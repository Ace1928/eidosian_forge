from __future__ import (absolute_import, division, print_function)
import copy
import json
from ansible_collections.theforeman.foreman.plugins.module_utils._version import LooseVersion
from time import sleep
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, to_safe_group_name, Constructable
def _populate_host_api(self):
    hostnames = self.get_option('hostnames')
    strict = self.get_option('strict')
    for host in self._get_hosts():
        if not host:
            continue
        composed_host_name = self._get_hostname(host, hostnames, strict=strict)
        if composed_host_name in self._cache.keys():
            continue
        host_name = self.inventory.add_host(composed_host_name)
        group_name = host.get('hostgroup_title', host.get('hostgroup_name'))
        if group_name:
            parent_name = None
            group_label_parts = []
            for part in group_name.split('/'):
                group_label_parts.append(part.lower().replace(' ', ''))
                gname = to_safe_group_name('%s%s' % (self.get_option('group_prefix'), '/'.join(group_label_parts)))
                result_gname = self.inventory.add_group(gname)
                if parent_name:
                    self.inventory.add_child(parent_name, result_gname)
                parent_name = result_gname
            self.inventory.add_child(result_gname, host_name)
        if self.get_option('legacy_hostvars'):
            hostvars = self._get_hostvars(host)
            self.inventory.set_variable(host_name, 'foreman', hostvars)
        else:
            omitted_vars = ('name', 'hostgroup_title', 'hostgroup_name')
            hostvars = self._get_hostvars(host, self.get_option('vars_prefix'), omitted_vars)
            for k, v in hostvars.items():
                try:
                    self.inventory.set_variable(host_name, k, v)
                except ValueError as e:
                    self.display.warning('Could not set host info hostvar for %s, skipping %s: %s' % (host, k, to_text(e)))
        if self.get_option('want_params'):
            params = self._get_all_params_by_id(host['id'])
            filtered_params = {}
            for p in params:
                if 'name' in p and 'value' in p:
                    filtered_params[p['name']] = p['value']
            if self.get_option('legacy_hostvars'):
                self.inventory.set_variable(host_name, 'foreman_params', filtered_params)
            else:
                for k, v in filtered_params.items():
                    try:
                        self.inventory.set_variable(host_name, k, v)
                    except ValueError as e:
                        self.display.warning("Could not set hostvar %s to '%s' for the '%s' host, skipping:  %s" % (k, to_native(v), host, to_native(e)))
        if self.get_option('want_facts'):
            self.inventory.set_variable(host_name, 'foreman_facts', self._get_facts(host))
        if self.get_option('want_hostcollections'):
            host_data = self._get_host_data_by_id(host['id'])
            hostcollections = host_data.get('host_collections')
            if hostcollections:
                for hostcollection in hostcollections:
                    try:
                        hostcollection_group = to_safe_group_name('%shostcollection_%s' % (self.get_option('group_prefix'), hostcollection['name'].lower().replace(' ', '')))
                        hostcollection_group = self.inventory.add_group(hostcollection_group)
                        self.inventory.add_child(hostcollection_group, host_name)
                    except ValueError as e:
                        self.display.warning('Could not create groups for host collections for %s, skipping: %s' % (host_name, to_text(e)))
        hostvars = self.inventory.get_host(host_name).get_vars()
        self._set_composite_vars(self.get_option('compose'), hostvars, host_name, strict)
        self._add_host_to_composed_groups(self.get_option('groups'), hostvars, host_name, strict)
        self._add_host_to_keyed_groups(self.get_option('keyed_groups'), hostvars, host_name, strict)