from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_color_hex(self, color_name):
    if color_name.startswith('#'):
        color_hex = color_name
    else:
        try:
            color_hex = webcolors.name_to_hex(color_name)
        except Exception as e:
            self._module.fail_json(msg="Failed to get RGB hex for color '%s': %s" % (color_name, e))
    color_hex = color_hex.strip('#').upper()
    return color_hex