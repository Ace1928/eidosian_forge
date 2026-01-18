from __future__ import absolute_import, division, print_function
import os
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils._stormssh import ConfigParser, HAS_PARAMIKO, PARAMIKO_IMPORT_ERROR
from ansible_collections.community.general.plugins.module_utils.ssh import determine_config_file
class SSHConfig(object):

    def __init__(self, module):
        self.module = module
        if not HAS_PARAMIKO:
            module.fail_json(msg=missing_required_lib('PARAMIKO'), exception=PARAMIKO_IMPORT_ERROR)
        self.params = module.params
        self.user = self.params.get('user')
        self.group = self.params.get('group') or self.user
        self.host = self.params.get('host')
        self.config_file = self.params.get('ssh_config_file')
        self.identity_file = self.params['identity_file']
        self.check_ssh_config_path()
        try:
            self.config = ConfigParser(self.config_file)
        except FileNotFoundError:
            self.module.fail_json(msg='Failed to find %s' % self.config_file)
        self.config.load()

    def check_ssh_config_path(self):
        self.config_file = determine_config_file(self.user, self.config_file)
        if os.path.exists(self.config_file) and self.identity_file is not None:
            dirname = os.path.dirname(self.config_file)
            self.identity_file = os.path.join(dirname, self.identity_file)
            if not os.path.exists(self.identity_file):
                self.module.fail_json(msg='IdentityFile %s does not exist' % self.params['identity_file'])

    def ensure_state(self):
        hosts_result = self.config.search_host(self.host)
        state = self.params['state']
        args = dict(hostname=self.params.get('hostname'), port=self.params.get('port'), identity_file=self.params.get('identity_file'), identities_only=convert_bool(self.params.get('identities_only')), user=self.params.get('remote_user'), strict_host_key_checking=self.params.get('strict_host_key_checking'), user_known_hosts_file=self.params.get('user_known_hosts_file'), proxycommand=self.params.get('proxycommand'), proxyjump=self.params.get('proxyjump'), host_key_algorithms=self.params.get('host_key_algorithms'), forward_agent=convert_bool(self.params.get('forward_agent')), add_keys_to_agent=convert_bool(self.params.get('add_keys_to_agent')), controlmaster=self.params.get('controlmaster'), controlpath=self.params.get('controlpath'), controlpersist=fix_bool_str(self.params.get('controlpersist')))
        config_changed = False
        hosts_changed = []
        hosts_change_diff = []
        hosts_removed = []
        hosts_added = []
        hosts_result = [host for host in hosts_result if host['host'] == self.host]
        if hosts_result:
            for host in hosts_result:
                if state == 'absent':
                    config_changed = True
                    hosts_removed.append(host['host'])
                    self.config.delete_host(host['host'])
                else:
                    changed, options = self.change_host(host['options'], **args)
                    if changed:
                        config_changed = True
                        self.config.update_host(host['host'], options)
                        hosts_changed.append(host['host'])
                        hosts_change_diff.append({host['host']: {'old': host['options'], 'new': options}})
        elif state == 'present':
            changed, options = self.change_host(dict(), **args)
            if changed:
                config_changed = True
                hosts_added.append(self.host)
                self.config.add_host(self.host, options)
        if config_changed and (not self.module.check_mode):
            try:
                self.config.write_to_ssh_config()
            except PermissionError as perm_exec:
                self.module.fail_json(msg='Failed to write to %s due to permission issue: %s' % (self.config_file, to_native(perm_exec)))
            perm_mode = '0600'
            if self.config_file == '/etc/ssh/ssh_config':
                perm_mode = '0644'
            self.module.set_mode_if_different(self.config_file, perm_mode, False)
            self.module.set_owner_if_different(self.config_file, self.user, False)
            self.module.set_group_if_different(self.config_file, self.group, False)
        self.module.exit_json(changed=config_changed, hosts_changed=hosts_changed, hosts_removed=hosts_removed, hosts_change_diff=hosts_change_diff, hosts_added=hosts_added)

    @staticmethod
    def change_host(options, **kwargs):
        options = deepcopy(options)
        changed = False
        for k, v in kwargs.items():
            if '_' in k:
                k = k.replace('_', '')
            if not v:
                if options.get(k):
                    del options[k]
                    changed = True
            elif options.get(k) != v and (not (isinstance(options.get(k), list) and v in options.get(k))):
                options[k] = v
                changed = True
        return (changed, options)