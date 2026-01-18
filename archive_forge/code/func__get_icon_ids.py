from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_icon_ids(self):
    icons_list = self._zapi.image.get({})
    icon_ids = {}
    for icon in icons_list:
        icon_ids[icon['name']] = icon['imageid']
    return icon_ids