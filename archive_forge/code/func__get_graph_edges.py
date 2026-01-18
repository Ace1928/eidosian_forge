from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_graph_edges(self, parent):
    edges = []
    for edge in parent.get_edges():
        edges.append(edge)
    for subgraph in parent.get_subgraphs():
        edges += self._get_graph_edges(subgraph)
    return edges