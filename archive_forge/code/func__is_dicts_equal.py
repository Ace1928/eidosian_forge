from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
@staticmethod
def _is_dicts_equal(d1, d2, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []
    for key in d1.keys():
        if isinstance(d1[key], dict) or isinstance(d1[key], list):
            continue
        if key in exclude_keys:
            continue
        if key not in d2 or str(d2[key]) != str(d1[key]):
            return False
    return True