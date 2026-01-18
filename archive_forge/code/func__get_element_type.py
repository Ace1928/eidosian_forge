from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_element_type(self, data):
    types = {'host': 0, 'sysmap': 1, 'trigger': 2, 'group': 3, 'image': 4}
    element_type = {'elementtype': types['image']}
    for type_name, type_id in sorted(types.items()):
        field_name = 'zbx_' + type_name
        if field_name in data:
            method_name = '_get_' + type_name + '_id'
            element_name = remove_quotes(data[field_name])
            get_element_id = getattr(self, method_name, None)
            if get_element_id:
                elementid = get_element_id(element_name)
                if elementid and int(elementid) > 0:
                    element_type.update({'elementtype': type_id, 'label': element_name})
                    element_type.update({'elements': [{type_name + 'id': elementid}]})
                    break
                else:
                    self._module.fail_json(msg="Failed to find id for %s '%s'" % (type_name, element_name))
    return element_type