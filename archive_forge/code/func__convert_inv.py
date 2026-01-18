from __future__ import (absolute_import, division, print_function)
import json
from ansible.errors import AnsibleParserError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _convert_inv(self, json_data):
    """Convert Icinga2 API data to JSON format for Ansible"""
    groups_dict = {'_meta': {'hostvars': {}}}
    for entry in json_data:
        host_attrs = make_unsafe(entry['attrs'])
        if self.inventory_attr == 'name':
            host_name = make_unsafe(entry.get('name'))
        if self.inventory_attr == 'address':
            if host_attrs.get('address', '') != '':
                host_name = make_unsafe(host_attrs.get('address'))
            else:
                host_name = make_unsafe(entry.get('name'))
        if self.inventory_attr == 'display_name':
            host_name = host_attrs.get('display_name')
        if host_attrs['state'] == 0:
            host_attrs['state'] = 'on'
        else:
            host_attrs['state'] = 'off'
        self.inventory.add_host(host_name)
        if self.group_by_hostgroups:
            host_groups = host_attrs.get('groups')
            for group in host_groups:
                if group not in self.inventory.groups.keys():
                    self.inventory.add_group(group)
                self.inventory.add_child(group, host_name)
        if host_attrs.get('address') != '':
            self.inventory.set_variable(host_name, 'ansible_host', host_attrs.get('address'))
        self.inventory.set_variable(host_name, 'hostname', make_unsafe(entry.get('name')))
        self.inventory.set_variable(host_name, 'display_name', host_attrs.get('display_name'))
        self.inventory.set_variable(host_name, 'state', host_attrs['state'])
        self.inventory.set_variable(host_name, 'state_type', host_attrs['state_type'])
        construct_vars = dict(self.inventory.get_host(host_name).get_vars())
        construct_vars['icinga2_attributes'] = host_attrs
        self._apply_constructable(host_name, construct_vars)
    return groups_dict