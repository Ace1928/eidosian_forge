from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_graph_nodes(self, parent):
    nodes = {}
    for node in parent.get_nodes():
        node_name = node.get_name()
        if node_name in ('node', 'graph', 'edge'):
            continue
        nodes[node_name] = self._get_node_attributes(node)
    for subgraph in parent.get_subgraphs():
        nodes.update(self._get_graph_nodes(subgraph))
    return nodes