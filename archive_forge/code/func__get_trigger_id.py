from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_trigger_id(self, trigger_definition):
    try:
        host, trigger = trigger_definition.split(':', 1)
    except Exception as e:
        self._module.fail_json(msg="Failed to parse zbx_trigger='%s': %s" % (trigger_definition, e))
    triggerid = self._zapi.trigger.get({'host': host, 'filter': {'description': trigger}})
    if triggerid:
        return str(triggerid[0]['triggerid'])