from __future__ import absolute_import, division, print_function
import copy
import datetime
import os
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
@staticmethod
def _instance_json_to_module_state(resp_json):
    if resp_json['type'] == 'error':
        return 'absent'
    return ANSIBLE_LXD_STATES[resp_json['metadata']['status']]