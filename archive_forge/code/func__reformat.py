import abc
from cliff.formatters import base
from networkx.drawing.nx_pydot import write_dot
from networkx.readwrite.graphml import GraphMLWriter
from networkx.readwrite import json_graph
import networkx as nx
@staticmethod
def _reformat(data):
    for node in data['nodes']:
        name = node.pop('name', None)
        v_type = node['vitrage_type']
        if name and name != v_type:
            node['label'] = name + '\n' + v_type
        else:
            node['label'] = v_type
        GraphFormatter._list2str(node)
    data['multigraph'] = False
    for node in data['links']:
        node['label'] = node.pop('relationship_type')
        node.pop('key')