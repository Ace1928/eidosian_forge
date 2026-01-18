from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
class DockerNetworkManager(object):

    def __init__(self, client):
        self.client = client
        self.parameters = TaskParameters(client)
        self.check_mode = self.client.check_mode
        self.results = {u'changed': False, u'actions': []}
        self.diff = self.client.module._diff
        self.diff_tracker = DifferenceTracker()
        self.diff_result = dict()
        self.existing_network = self.get_existing_network()
        if not self.parameters.connected and self.existing_network:
            self.parameters.connected = container_names_in_network(self.existing_network)
        if self.parameters.ipam_config:
            try:
                for ipam_config in self.parameters.ipam_config:
                    validate_cidr(ipam_config['subnet'])
            except ValueError as e:
                self.client.fail(to_native(e))
        if self.parameters.driver_options:
            self.parameters.driver_options = clean_dict_booleans_for_docker_api(self.parameters.driver_options)
        state = self.parameters.state
        if state == 'present':
            self.present()
        elif state == 'absent':
            self.absent()
        if self.diff or self.check_mode or self.parameters.debug:
            if self.diff:
                self.diff_result['before'], self.diff_result['after'] = self.diff_tracker.get_before_after()
            self.results['diff'] = self.diff_result

    def get_existing_network(self):
        return self.client.get_network(name=self.parameters.name)

    def has_different_config(self, net):
        """
        Evaluates an existing network and returns a tuple containing a boolean
        indicating if the configuration is different and a list of differences.

        :param net: the inspection output for an existing network
        :return: (bool, list)
        """
        differences = DifferenceTracker()
        if self.parameters.driver and self.parameters.driver != net['Driver']:
            differences.add('driver', parameter=self.parameters.driver, active=net['Driver'])
        if self.parameters.driver_options:
            if not net.get('Options'):
                differences.add('driver_options', parameter=self.parameters.driver_options, active=net.get('Options'))
            else:
                for key, value in self.parameters.driver_options.items():
                    if not key in net['Options'] or value != net['Options'][key]:
                        differences.add('driver_options.%s' % key, parameter=value, active=net['Options'].get(key))
        if self.parameters.ipam_driver:
            if not net.get('IPAM') or net['IPAM']['Driver'] != self.parameters.ipam_driver:
                differences.add('ipam_driver', parameter=self.parameters.ipam_driver, active=net.get('IPAM'))
        if self.parameters.ipam_driver_options is not None:
            ipam_driver_options = net['IPAM'].get('Options') or {}
            if ipam_driver_options != self.parameters.ipam_driver_options:
                differences.add('ipam_driver_options', parameter=self.parameters.ipam_driver_options, active=ipam_driver_options)
        if self.parameters.ipam_config is not None and self.parameters.ipam_config:
            if not net.get('IPAM') or not net['IPAM']['Config']:
                differences.add('ipam_config', parameter=self.parameters.ipam_config, active=net.get('IPAM', {}).get('Config'))
            else:
                net_ipam_configs = []
                for net_ipam_config in net['IPAM']['Config']:
                    config = dict()
                    for k, v in net_ipam_config.items():
                        config[normalize_ipam_config_key(k)] = v
                    net_ipam_configs.append(config)
                for idx, ipam_config in enumerate(self.parameters.ipam_config):
                    net_config = dict()
                    for net_ipam_config in net_ipam_configs:
                        if dicts_are_essentially_equal(ipam_config, net_ipam_config):
                            net_config = net_ipam_config
                            break
                    for key, value in ipam_config.items():
                        if value is None:
                            continue
                        if value != net_config.get(key):
                            differences.add('ipam_config[%s].%s' % (idx, key), parameter=value, active=net_config.get(key))
        if self.parameters.enable_ipv6 is not None and self.parameters.enable_ipv6 != net.get('EnableIPv6', False):
            differences.add('enable_ipv6', parameter=self.parameters.enable_ipv6, active=net.get('EnableIPv6', False))
        if self.parameters.internal is not None and self.parameters.internal != net.get('Internal', False):
            differences.add('internal', parameter=self.parameters.internal, active=net.get('Internal'))
        if self.parameters.scope is not None and self.parameters.scope != net.get('Scope'):
            differences.add('scope', parameter=self.parameters.scope, active=net.get('Scope'))
        if self.parameters.attachable is not None and self.parameters.attachable != net.get('Attachable', False):
            differences.add('attachable', parameter=self.parameters.attachable, active=net.get('Attachable'))
        if self.parameters.labels:
            if not net.get('Labels'):
                differences.add('labels', parameter=self.parameters.labels, active=net.get('Labels'))
            else:
                for key, value in self.parameters.labels.items():
                    if not key in net['Labels'] or value != net['Labels'][key]:
                        differences.add('labels.%s' % key, parameter=value, active=net['Labels'].get(key))
        return (not differences.empty, differences)

    def create_network(self):
        if not self.existing_network:
            data = {'Name': self.parameters.name, 'Driver': self.parameters.driver, 'Options': self.parameters.driver_options, 'IPAM': None, 'CheckDuplicate': None}
            if self.parameters.enable_ipv6:
                data['EnableIPv6'] = True
            if self.parameters.internal:
                data['Internal'] = True
            if self.parameters.scope is not None:
                data['Scope'] = self.parameters.scope
            if self.parameters.attachable is not None:
                data['Attachable'] = self.parameters.attachable
            if self.parameters.labels is not None:
                data['Labels'] = self.parameters.labels
            ipam_pools = []
            if self.parameters.ipam_config:
                for ipam_pool in self.parameters.ipam_config:
                    ipam_pools.append({'Subnet': ipam_pool['subnet'], 'IPRange': ipam_pool['iprange'], 'Gateway': ipam_pool['gateway'], 'AuxiliaryAddresses': ipam_pool['aux_addresses']})
            if self.parameters.ipam_driver or self.parameters.ipam_driver_options or ipam_pools:
                data['IPAM'] = {'Driver': self.parameters.ipam_driver, 'Config': ipam_pools or [], 'Options': self.parameters.ipam_driver_options}
            if not self.check_mode:
                resp = self.client.post_json_to_json('/networks/create', data=data)
                self.client.report_warnings(resp, ['Warning'])
                self.existing_network = self.client.get_network(network_id=resp['Id'])
            self.results['actions'].append('Created network %s with driver %s' % (self.parameters.name, self.parameters.driver))
            self.results['changed'] = True

    def remove_network(self):
        if self.existing_network:
            self.disconnect_all_containers()
            if not self.check_mode:
                self.client.delete_call('/networks/{0}', self.parameters.name)
            self.results['actions'].append('Removed network %s' % (self.parameters.name,))
            self.results['changed'] = True

    def is_container_connected(self, container_name):
        if not self.existing_network:
            return False
        return container_name in container_names_in_network(self.existing_network)

    def connect_containers(self):
        for name in self.parameters.connected:
            if not self.is_container_connected(name):
                if not self.check_mode:
                    data = {'Container': name, 'EndpointConfig': None}
                    self.client.post_json('/networks/{0}/connect', self.parameters.name, data=data)
                self.results['actions'].append('Connected container %s' % (name,))
                self.results['changed'] = True
                self.diff_tracker.add('connected.{0}'.format(name), parameter=True, active=False)

    def disconnect_missing(self):
        if not self.existing_network:
            return
        containers = self.existing_network['Containers']
        if not containers:
            return
        for c in containers.values():
            name = c['Name']
            if name not in self.parameters.connected:
                self.disconnect_container(name)

    def disconnect_all_containers(self):
        containers = self.client.get_network(name=self.parameters.name)['Containers']
        if not containers:
            return
        for cont in containers.values():
            self.disconnect_container(cont['Name'])

    def disconnect_container(self, container_name):
        if not self.check_mode:
            data = {'Container': container_name}
            self.client.post_json('/networks/{0}/disconnect', self.parameters.name, data=data)
        self.results['actions'].append('Disconnected container %s' % (container_name,))
        self.results['changed'] = True
        self.diff_tracker.add('connected.{0}'.format(container_name), parameter=False, active=True)

    def present(self):
        different = False
        differences = DifferenceTracker()
        if self.existing_network:
            different, differences = self.has_different_config(self.existing_network)
        self.diff_tracker.add('exists', parameter=True, active=self.existing_network is not None)
        if self.parameters.force or different:
            self.remove_network()
            self.existing_network = None
        self.create_network()
        self.connect_containers()
        if not self.parameters.appends:
            self.disconnect_missing()
        if self.diff or self.check_mode or self.parameters.debug:
            self.diff_result['differences'] = differences.get_legacy_docker_diffs()
            self.diff_tracker.merge(differences)
        if not self.check_mode and (not self.parameters.debug):
            self.results.pop('actions')
        network_facts = self.get_existing_network()
        self.results['network'] = network_facts

    def absent(self):
        self.diff_tracker.add('exists', parameter=False, active=self.existing_network is not None)
        self.remove_network()