from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def __find_dict(self, outval, inval):
    for out in outval:
        m = True
        for k, v in inval.items():
            if out[k] == str(v):
                continue
            else:
                m = False
        if m:
            break
    return m