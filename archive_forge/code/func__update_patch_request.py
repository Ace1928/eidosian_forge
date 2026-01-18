from __future__ import absolute_import, division, print_function
import json
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import ConfigBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, dict_diff
from ansible_collections.community.network.plugins.module_utils.network.exos.facts.facts import Facts
from ansible_collections.community.network.plugins.module_utils.network.exos.exos import send_requests
def _update_patch_request(self, want, have):
    facts, _warnings = Facts(self._module).get_facts(self.gather_subset, ['vlans'])
    vlans_facts = facts['ansible_network_resources'].get('vlans')
    vlan_id = []
    for vlan in vlans_facts:
        vlan_id.append(vlan['vlan_id'])
    if want.get('access'):
        if want['access']['vlan'] in vlan_id:
            l2_request = deepcopy(self.L2_INTERFACE_ACCESS)
            l2_request['data']['openconfig-vlan:config']['access-vlan'] = want['access']['vlan']
            l2_request['path'] = self.L2_PATH + str(want['name']) + '/openconfig-if-ethernet:ethernet/openconfig-vlan:switched-vlan/config'
        else:
            self._module.fail_json(msg='VLAN %s does not exist' % want['access']['vlan'])
    elif want.get('trunk'):
        if want['trunk']['native_vlan']:
            if want['trunk']['native_vlan'] in vlan_id:
                l2_request = deepcopy(self.L2_INTERFACE_NATIVE)
                l2_request['data']['openconfig-vlan:config']['native-vlan'] = want['trunk']['native_vlan']
                l2_request['path'] = self.L2_PATH + str(want['name']) + '/openconfig-if-ethernet:ethernet/openconfig-vlan:switched-vlan/config'
                for vlan in want['trunk']['trunk_allowed_vlans']:
                    if int(vlan) in vlan_id:
                        l2_request['data']['openconfig-vlan:config']['trunk-vlans'].append(int(vlan))
                    else:
                        self._module.fail_json(msg='VLAN %s does not exist' % vlan)
            else:
                self._module.fail_json(msg='VLAN %s does not exist' % want['trunk']['native_vlan'])
        else:
            l2_request = deepcopy(self.L2_INTERFACE_TRUNK)
            l2_request['path'] = self.L2_PATH + str(want['name']) + '/openconfig-if-ethernet:ethernet/openconfig-vlan:switched-vlan/config'
            for vlan in want['trunk']['trunk_allowed_vlans']:
                if int(vlan) in vlan_id:
                    l2_request['data']['openconfig-vlan:config']['trunk-vlans'].append(int(vlan))
                else:
                    self._module.fail_json(msg='VLAN %s does not exist' % vlan)
    return l2_request