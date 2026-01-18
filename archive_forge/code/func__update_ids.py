from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _update_ids(self, generated_map_config, exist_map_config):
    generated_selements_sorted = sorted(generated_map_config['selements'], key=itemgetter(*self.selements_sort_keys))
    exist_selements_sorted = sorted(exist_map_config['selements'], key=itemgetter(*self.selements_sort_keys))
    id_mapping = {}
    for generated_selement, exist_selement in zip(generated_selements_sorted, exist_selements_sorted):
        id_mapping[exist_selement['selementid']] = generated_selement['selementid']
    for link in exist_map_config['links']:
        link['selementid1'] = id_mapping[link['selementid1']]
        link['selementid2'] = id_mapping[link['selementid2']]
        if link['selementid2'] < link['selementid1']:
            link['selementid1'], link['selementid2'] = (link['selementid2'], link['selementid1'])