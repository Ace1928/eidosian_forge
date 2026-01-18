from __future__ import absolute_import, division, print_function
import json
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import ConfigBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, dict_diff
from ansible_collections.community.network.plugins.module_utils.network.exos.facts.facts import Facts
from ansible_collections.community.network.plugins.module_utils.network.exos.exos import send_requests
def _update_delete_request(self, have):
    l2_request = deepcopy(self.L2_INTERFACE_ACCESS)
    if have['access'] and have['access']['vlan'] != 1 or have['trunk'] or (not have['access']):
        l2_request['data']['openconfig-vlan:config']['access-vlan'] = 1
        l2_request['path'] = self.L2_PATH + str(have['name']) + '/openconfig-if-ethernet:ethernet/openconfig-vlan:switched-vlan/config'
    return l2_request