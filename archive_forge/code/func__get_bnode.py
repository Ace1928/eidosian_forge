from datetime import datetime
from prov.graph import INFERRED_ELEMENT_CLASS
from prov.model import (
import pydot
def _get_bnode():
    count[1] += 1
    bnode_id = 'b%d' % count[1]
    bnode = pydot.Node(bnode_id, label='""', shape='point', color='gray')
    dot.add_node(bnode)
    return bnode