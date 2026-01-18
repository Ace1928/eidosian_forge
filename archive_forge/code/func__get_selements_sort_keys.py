from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_selements_sort_keys(self):
    keys_to_sort = ['label']
    return keys_to_sort