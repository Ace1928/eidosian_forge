from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_links(self, nodes, edges):
    links = {}
    for edge in edges:
        link_id = tuple(sorted(edge.obj_dict['points']))
        node1, node2 = link_id
        data = edge.obj_dict['attributes']
        if 'style' in data and data['style'] == 'invis':
            continue
        if link_id not in links:
            links[link_id] = {'selementid1': min(nodes[node1]['selementid'], nodes[node2]['selementid']), 'selementid2': max(nodes[node1]['selementid'], nodes[node2]['selementid'])}
        link = links[link_id]
        if 'color' not in link:
            link['color'] = self._get_color_hex(remove_quotes(data.get('color', 'green')))
        if 'zbx_draw_style' not in link:
            link['drawtype'] = self._get_link_draw_style_id(remove_quotes(data.get('zbx_draw_style', 'line')))
        label = self._get_label(data)
        if label and 'label' not in link:
            link['label'] = label
        triggers = self._get_triggers(data)
        if triggers:
            if 'linktriggers' not in link:
                link['linktriggers'] = []
            link['linktriggers'] += triggers
    return list(links.values())