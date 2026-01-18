from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_link_draw_style_id(self, draw_style):
    draw_style_ids = {'line': 0, 'bold': 2, 'dotted': 3, 'dashed': 4}
    try:
        draw_style_id = draw_style_ids[draw_style]
    except Exception as e:
        self._module.fail_json(msg="Failed to find id for draw type '%s': %s" % (draw_style, e))
    return draw_style_id