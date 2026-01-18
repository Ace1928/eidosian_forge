from __future__ import absolute_import, division, print_function
import collections
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
class CliConfiguration(ConfigBase):

    def __init__(self, module):
        super(CliConfiguration, self).__init__(module)

    def map_obj_to_commands(self):
        commands = list()
        state = self._module.params['state']

        def needs_update(x):
            return self._want.get(x) and self._want.get(x) != self._have.get(x)
        if state == 'absent':
            if self._have['hostname'] != 'ios':
                commands.append('no hostname')
            if self._have['domain_name']:
                commands.append('no domain name')
            if self._have['lookup_source']:
                commands.append('no domain lookup source-interface {0!s}'.format(self._have['lookup_source']))
            if not self._have['lookup_enabled']:
                commands.append('no domain lookup disable')
            for item in self._have['name_servers']:
                commands.append('no domain name-server {0!s}'.format(item))
            for item in self._have['domain_search']:
                commands.append('no domain list {0!s}'.format(item))
        elif state == 'present':
            if needs_update('hostname'):
                commands.append('hostname {0!s}'.format(self._want['hostname']))
            if needs_update('domain_name'):
                commands.append('domain name {0!s}'.format(self._want['domain_name']))
            if needs_update('lookup_source'):
                commands.append('domain lookup source-interface {0!s}'.format(self._want['lookup_source']))
            cmd = None
            if not self._want['lookup_enabled'] and self._have['lookup_enabled']:
                cmd = 'domain lookup disable'
            elif self._want['lookup_enabled'] and (not self._have['lookup_enabled']):
                cmd = 'no domain lookup disable'
            if cmd is not None:
                commands.append(cmd)
            if self._want['name_servers'] is not None:
                adds, removes = diff_list(self._want['name_servers'], self._have['name_servers'])
                for item in adds:
                    commands.append('domain name-server {0!s}'.format(item))
                for item in removes:
                    commands.append('no domain name-server {0!s}'.format(item))
            if self._want['domain_search'] is not None:
                adds, removes = diff_list(self._want['domain_search'], self._have['domain_search'])
                for item in adds:
                    commands.append('domain list {0!s}'.format(item))
                for item in removes:
                    commands.append('no domain list {0!s}'.format(item))
        self._result['commands'] = []
        if commands:
            commit = not self._module.check_mode
            diff = load_config(self._module, commands, commit=commit)
            if diff:
                self._result['diff'] = dict(prepared=diff)
            self._result['commands'] = commands
            self._result['changed'] = True

    def parse_hostname(self, config):
        match = re.search('^hostname (\\S+)', config, re.M)
        if match:
            return match.group(1)

    def parse_domain_name(self, config):
        match = re.search('^domain name (\\S+)', config, re.M)
        if match:
            return match.group(1)

    def parse_lookup_source(self, config):
        match = re.search('^domain lookup source-interface (\\S+)', config, re.M)
        if match:
            return match.group(1)

    def map_config_to_obj(self):
        config = get_config(self._module)
        self._have.update({'hostname': self.parse_hostname(config), 'domain_name': self.parse_domain_name(config), 'domain_search': re.findall('^domain list (\\S+)', config, re.M), 'lookup_source': self.parse_lookup_source(config), 'lookup_enabled': 'domain lookup disable' not in config, 'name_servers': re.findall('^domain name-server (\\S+)', config, re.M)})

    def run(self):
        self.map_params_to_obj()
        self.map_config_to_obj()
        self.map_obj_to_commands()
        return self._result