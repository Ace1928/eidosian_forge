from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.service import (
def _service_list_to_dict(self, data):
    """Convert all list of dicts to dicts of dicts"""
    p_key = {'timestamps': 'msg'}
    tmp_data = deepcopy(data)
    for k, _v in p_key.items():
        if k in tmp_data:
            tmp_data[k] = {str(i[p_key.get(k)]): i for i in tmp_data[k]}
    return tmp_data