from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_label_type_id(self, label_type):
    label_type_ids = {'label': 0, 'ip': 1, 'name': 2, 'status': 3, 'nothing': 4, 'custom': 5}
    try:
        label_type_id = label_type_ids[label_type]
    except Exception as e:
        self._module.fail_json(msg="Failed to find id for label type '%s': %s" % (label_type, e))
    return label_type_id