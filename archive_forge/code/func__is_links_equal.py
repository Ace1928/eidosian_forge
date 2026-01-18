from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _is_links_equal(self, generated_links, exist_links):
    if len(generated_links) != len(exist_links):
        return False
    generated_links_sorted = sorted(generated_links, key=itemgetter('selementid1', 'selementid2', 'color', 'drawtype'))
    exist_links_sorted = sorted(exist_links, key=itemgetter('selementid1', 'selementid2', 'color', 'drawtype'))
    for generated_link, exist_link in zip(generated_links_sorted, exist_links_sorted):
        if not self._is_dicts_equal(generated_link, exist_link, ['selementid1', 'selementid2']):
            return False
        if not self._is_triggers_equal(generated_link.get('linktriggers', []), exist_link.get('linktriggers', [])):
            return False
    return True