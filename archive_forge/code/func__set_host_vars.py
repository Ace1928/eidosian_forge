from __future__ import (absolute_import, division, print_function)
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
def _set_host_vars(self, host):
    self.inventory.set_variable(host, 'idrac_ip', host)
    self.inventory.set_variable(host, 'baseuri', host)
    self.inventory.set_variable(host, 'hostname', host)
    if 'host_vars' in self.config:
        host_vars = self.get_option('host_vars')
        for key, val in dict(host_vars).items():
            self.inventory.set_variable(host, key, val)