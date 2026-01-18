from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_icon_size(self, icon_id):
    icons_list = self._zapi.image.get({'imageids': [icon_id], 'select_image': True})
    if len(icons_list) > 0:
        icon_base64 = icons_list[0]['image']
    else:
        self._module.fail_json(msg='Failed to find image with id %s' % icon_id)
    image = Image.open(BytesIO(base64.b64decode(icon_base64)))
    icon_width, icon_height = image.size
    return (icon_width, icon_height)