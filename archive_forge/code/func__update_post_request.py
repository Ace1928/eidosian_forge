from __future__ import absolute_import, division, print_function
import json
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import ConfigBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, dict_diff
from ansible_collections.community.network.plugins.module_utils.network.exos.facts.facts import Facts
from ansible_collections.community.network.plugins.module_utils.network.exos.exos import send_requests
from ansible_collections.community.network.plugins.module_utils.network.exos.utils.utils import search_obj_in_list
def _update_post_request(self, want):
    request_post = deepcopy(self.VLAN_POST)
    request_body = deepcopy(self.REQUEST_BODY)
    request_body = self._update_vlan_config_body(want, request_body)
    request_post['data']['openconfig-vlan:vlans'].append(request_body)
    request_post['data'] = json.dumps(request_post['data'])
    return request_post