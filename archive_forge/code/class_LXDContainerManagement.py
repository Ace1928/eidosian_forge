from __future__ import absolute_import, division, print_function
import copy
import datetime
import os
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
class LXDContainerManagement(object):

    def __init__(self, module):
        """Management of LXC containers via Ansible.

        :param module: Processed Ansible Module.
        :type module: ``object``
        """
        self.module = module
        self.name = self.module.params['name']
        self.project = self.module.params['project']
        self._build_config()
        self.state = self.module.params['state']
        self.timeout = self.module.params['timeout']
        self.wait_for_ipv4_addresses = self.module.params['wait_for_ipv4_addresses']
        self.force_stop = self.module.params['force_stop']
        self.addresses = None
        self.target = self.module.params['target']
        self.wait_for_container = self.module.params['wait_for_container']
        self.type = self.module.params['type']
        self.key_file = self.module.params.get('client_key')
        if self.key_file is None:
            self.key_file = '{0}/.config/lxc/client.key'.format(os.environ['HOME'])
        self.cert_file = self.module.params.get('client_cert')
        if self.cert_file is None:
            self.cert_file = '{0}/.config/lxc/client.crt'.format(os.environ['HOME'])
        self.debug = self.module._verbosity >= 4
        try:
            if self.module.params['url'] != ANSIBLE_LXD_DEFAULT_URL:
                self.url = self.module.params['url']
            elif os.path.exists(self.module.params['snap_url'].replace('unix:', '')):
                self.url = self.module.params['snap_url']
            else:
                self.url = self.module.params['url']
        except Exception as e:
            self.module.fail_json(msg=e.msg)
        try:
            self.client = LXDClient(self.url, key_file=self.key_file, cert_file=self.cert_file, debug=self.debug)
        except LXDClientException as e:
            self.module.fail_json(msg=e.msg)
        self.api_endpoint = '/1.0/instances'
        check_api_endpoint = self.client.do('GET', '{0}?project='.format(self.api_endpoint), ok_error_codes=[404])
        if check_api_endpoint['error_code'] == 404:
            if self.type == 'container':
                self.api_endpoint = '/1.0/containers'
            elif self.type == 'virtual-machine':
                self.api_endpoint = '/1.0/virtual-machines'
        self.trust_password = self.module.params.get('trust_password', None)
        self.actions = []
        self.diff = {'before': {}, 'after': {}}
        self.old_instance_json = {}
        self.old_sections = {}

    def _build_config(self):
        self.config = {}
        for attr in CONFIG_PARAMS:
            param_val = self.module.params.get(attr, None)
            if param_val is not None:
                self.config[attr] = param_val

    def _get_instance_json(self):
        url = '{0}/{1}'.format(self.api_endpoint, self.name)
        if self.project:
            url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
        return self.client.do('GET', url, ok_error_codes=[404])

    def _get_instance_state_json(self):
        url = '{0}/{1}/state'.format(self.api_endpoint, self.name)
        if self.project:
            url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
        return self.client.do('GET', url, ok_error_codes=[404])

    @staticmethod
    def _instance_json_to_module_state(resp_json):
        if resp_json['type'] == 'error':
            return 'absent'
        return ANSIBLE_LXD_STATES[resp_json['metadata']['status']]

    def _change_state(self, action, force_stop=False):
        url = '{0}/{1}/state'.format(self.api_endpoint, self.name)
        if self.project:
            url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
        body_json = {'action': action, 'timeout': self.timeout}
        if force_stop:
            body_json['force'] = True
        if not self.module.check_mode:
            return self.client.do('PUT', url, body_json=body_json)

    def _create_instance(self):
        url = self.api_endpoint
        url_params = dict()
        if self.target:
            url_params['target'] = self.target
        if self.project:
            url_params['project'] = self.project
        if url_params:
            url = '{0}?{1}'.format(url, urlencode(url_params))
        config = self.config.copy()
        config['name'] = self.name
        if self.type not in self.api_endpoint:
            config['type'] = self.type
        if not self.module.check_mode:
            self.client.do('POST', url, config, wait_for_container=self.wait_for_container)
        self.actions.append('create')

    def _start_instance(self):
        self._change_state('start')
        self.actions.append('start')

    def _stop_instance(self):
        self._change_state('stop', self.force_stop)
        self.actions.append('stop')

    def _restart_instance(self):
        self._change_state('restart', self.force_stop)
        self.actions.append('restart')

    def _delete_instance(self):
        url = '{0}/{1}'.format(self.api_endpoint, self.name)
        if self.project:
            url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
        if not self.module.check_mode:
            self.client.do('DELETE', url)
        self.actions.append('delete')

    def _freeze_instance(self):
        self._change_state('freeze')
        self.actions.append('freeze')

    def _unfreeze_instance(self):
        self._change_state('unfreeze')
        self.actions.append('unfreeze')

    def _instance_ipv4_addresses(self, ignore_devices=None):
        ignore_devices = ['lo'] if ignore_devices is None else ignore_devices
        data = (self._get_instance_state_json() or {}).get('metadata', None) or {}
        network = dict(((k, v) for k, v in (data.get('network', None) or {}).items() if k not in ignore_devices))
        addresses = dict(((k, [a['address'] for a in v['addresses'] if a['family'] == 'inet']) for k, v in network.items()))
        return addresses

    @staticmethod
    def _has_all_ipv4_addresses(addresses):
        return len(addresses) > 0 and all((len(v) > 0 for v in addresses.values()))

    def _get_addresses(self):
        try:
            due = datetime.datetime.now() + datetime.timedelta(seconds=self.timeout)
            while datetime.datetime.now() < due:
                time.sleep(1)
                addresses = self._instance_ipv4_addresses()
                if self._has_all_ipv4_addresses(addresses) or self.module.check_mode:
                    self.addresses = addresses
                    return
        except LXDClientException as e:
            e.msg = 'timeout for getting IPv4 addresses'
            raise

    def _started(self):
        if self.old_state == 'absent':
            self._create_instance()
            self._start_instance()
        else:
            if self.old_state == 'frozen':
                self._unfreeze_instance()
            elif self.old_state == 'stopped':
                self._start_instance()
            if self._needs_to_apply_instance_configs():
                self._apply_instance_configs()
        if self.wait_for_ipv4_addresses:
            self._get_addresses()

    def _stopped(self):
        if self.old_state == 'absent':
            self._create_instance()
        elif self.old_state == 'stopped':
            if self._needs_to_apply_instance_configs():
                self._start_instance()
                self._apply_instance_configs()
                self._stop_instance()
        else:
            if self.old_state == 'frozen':
                self._unfreeze_instance()
            if self._needs_to_apply_instance_configs():
                self._apply_instance_configs()
            self._stop_instance()

    def _restarted(self):
        if self.old_state == 'absent':
            self._create_instance()
            self._start_instance()
        else:
            if self.old_state == 'frozen':
                self._unfreeze_instance()
            if self._needs_to_apply_instance_configs():
                self._apply_instance_configs()
            self._restart_instance()
        if self.wait_for_ipv4_addresses:
            self._get_addresses()

    def _destroyed(self):
        if self.old_state != 'absent':
            if self.old_state == 'frozen':
                self._unfreeze_instance()
            if self.old_state != 'stopped':
                self._stop_instance()
            self._delete_instance()

    def _frozen(self):
        if self.old_state == 'absent':
            self._create_instance()
            self._start_instance()
            self._freeze_instance()
        else:
            if self.old_state == 'stopped':
                self._start_instance()
            if self._needs_to_apply_instance_configs():
                self._apply_instance_configs()
            self._freeze_instance()

    def _needs_to_change_instance_config(self, key):
        if key not in self.config:
            return False
        if key == 'config':
            old_configs = dict(self.old_sections.get(key, None) or {})
            for k, v in self.config['config'].items():
                if k not in old_configs:
                    return True
                if old_configs[k] != v:
                    return True
            return False
        else:
            old_configs = self.old_sections.get(key, {})
            return self.config[key] != old_configs

    def _needs_to_apply_instance_configs(self):
        for param in set(CONFIG_PARAMS) - set(CONFIG_CREATION_PARAMS):
            if self._needs_to_change_instance_config(param):
                return True
        return False

    def _apply_instance_configs(self):
        old_metadata = copy.deepcopy(self.old_instance_json).get('metadata', None) or {}
        body_json = {}
        for param in set(CONFIG_PARAMS) - set(CONFIG_CREATION_PARAMS):
            if param in old_metadata:
                body_json[param] = old_metadata[param]
            if self._needs_to_change_instance_config(param):
                if param == 'config':
                    body_json['config'] = body_json.get('config', None) or {}
                    for k, v in self.config['config'].items():
                        body_json['config'][k] = v
                else:
                    body_json[param] = self.config[param]
        self.diff['after']['instance'] = body_json
        url = '{0}/{1}'.format(self.api_endpoint, self.name)
        if self.project:
            url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
        if not self.module.check_mode:
            self.client.do('PUT', url, body_json=body_json)
        self.actions.append('apply_instance_configs')

    def run(self):
        """Run the main method."""
        try:
            if self.trust_password is not None:
                self.client.authenticate(self.trust_password)
            self.ignore_volatile_options = self.module.params.get('ignore_volatile_options')
            self.old_instance_json = self._get_instance_json()
            self.old_sections = dict(((section, content) if not isinstance(content, dict) else (section, dict(((k, v) for k, v in content.items() if not (self.ignore_volatile_options and k.startswith('volatile.'))))) for section, content in (self.old_instance_json.get('metadata', None) or {}).items() if section in set(CONFIG_PARAMS) - set(CONFIG_CREATION_PARAMS)))
            self.diff['before']['instance'] = self.old_sections
            self.diff['after']['instance'] = self.config
            self.old_state = self._instance_json_to_module_state(self.old_instance_json)
            self.diff['before']['state'] = self.old_state
            self.diff['after']['state'] = self.state
            action = getattr(self, LXD_ANSIBLE_STATES[self.state])
            action()
            state_changed = len(self.actions) > 0
            result_json = {'log_verbosity': self.module._verbosity, 'changed': state_changed, 'old_state': self.old_state, 'actions': self.actions, 'diff': self.diff}
            if self.client.debug:
                result_json['logs'] = self.client.logs
            if self.addresses is not None:
                result_json['addresses'] = self.addresses
            self.module.exit_json(**result_json)
        except LXDClientException as e:
            state_changed = len(self.actions) > 0
            fail_params = {'msg': e.msg, 'changed': state_changed, 'actions': self.actions, 'diff': self.diff}
            if self.client.debug:
                fail_params['logs'] = e.kwargs['logs']
            self.module.fail_json(**fail_params)