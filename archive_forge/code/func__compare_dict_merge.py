from __future__ import (absolute_import, division, print_function)
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def _compare_dict_merge(src_dict, new_dict, param_list):
    diff = 0
    for parm in param_list:
        val = new_dict.get(parm)
        if val is not None:
            if val != src_dict.get(parm):
                src_dict[parm] = val
                diff += 1
    return diff