from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_scales(self, graph):
    bb = remove_quotes(graph.get_bb())
    min_x, min_y, max_x, max_y = bb.split(',')
    scale_x = (self.width - self.margin * 2) / (float(max_x) - float(min_x)) if float(max_x) != float(min_x) else 0
    scale_y = (self.height - self.margin * 2) / (float(max_y) - float(min_y)) if float(max_y) != float(min_y) else 0
    return {'min_x': float(min_x), 'min_y': float(min_y), 'max_x': float(max_x), 'max_y': float(max_y), 'scale_x': float(scale_x), 'scale_y': float(scale_y)}