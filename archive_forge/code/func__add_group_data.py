from __future__ import (absolute_import, division, print_function)
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
def _add_group_data(self, group_data):
    visible_gdata = list(filter(lambda d: d.get('Visible') in [False], group_data))
    if visible_gdata:
        for gp in visible_gdata:
            group_data.remove(gp)
    for gdata in group_data:
        self._set_group_vars(gdata['Name'])
        device_ip = self._get_all_devices(gdata['AllLeafDevices@odata.navigationLink'])
        for hst in device_ip:
            self.inventory.add_host(host=hst, group=gdata['Name'])
            self._set_host_vars(hst)
    self._set_child_group(group_data)