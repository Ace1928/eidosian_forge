from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_triggers(self, data):
    triggers = []
    for trigger_definition in [remove_quotes(value) for key, value in data.items() if key.startswith('zbx_trigger')]:
        triggerid = self._get_trigger_id(trigger_definition)
        if triggerid:
            triggers.append({'triggerid': triggerid, 'color': self._get_color_hex(remove_quotes(data.get('zbx_trigger_color', 'red'))), 'drawtype': self._get_link_draw_style_id(remove_quotes(data.get('zbx_trigger_draw_style', 'bold')))})
        else:
            self._module.fail_json(msg="Failed to find trigger '%s'" % trigger_definition)
    return triggers