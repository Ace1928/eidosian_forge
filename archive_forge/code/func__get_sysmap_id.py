from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_sysmap_id(self, map_name):
    exist_map = self._zapi.map.get({'filter': {'name': map_name}})
    if exist_map:
        return exist_map[0]['sysmapid']
    return None