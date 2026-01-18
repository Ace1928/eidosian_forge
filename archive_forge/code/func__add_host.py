import collections
import sys
from ansible.errors import AnsibleParserError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
def _add_host(self, hostname, host_vars):
    self.inventory.add_host(hostname, group='all')
    for k, v in host_vars.items():
        self.inventory.set_variable(hostname, k, v)
    strict = self.get_option('strict')
    self._set_composite_vars(self.get_option('compose'), host_vars, hostname, strict=True)
    self._add_host_to_composed_groups(self.get_option('groups'), host_vars, hostname, strict=strict)
    self._add_host_to_keyed_groups(self.get_option('keyed_groups'), host_vars, hostname, strict=strict)