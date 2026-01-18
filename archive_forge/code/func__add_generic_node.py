from datetime import datetime
from prov.graph import INFERRED_ELEMENT_CLASS
from prov.model import (
import pydot
def _add_generic_node(qname, prov_type=None):
    count[0] += 1
    node_id = 'n%d' % count[0]
    node_label = f'"{qname}"'
    uri = qname.uri
    style = GENERIC_NODE_STYLE[prov_type] if prov_type else DOT_PROV_STYLE[0]
    node = pydot.Node(node_id, label=node_label, URL='"%s"' % uri, **style)
    node_map[uri] = node
    dot.add_node(node)
    return node