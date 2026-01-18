import threading
import fasteners
from oslo_utils import excutils
from taskflow import flow
from taskflow import logging
from taskflow import task
from taskflow.types import graph as gr
from taskflow.types import tree as tr
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.flow import (LINK_INVARIANT, LINK_RETRY)  # noqa
def _add_update_edges(graph, nodes_from, nodes_to, attr_dict=None):
    """Adds/updates edges from nodes to other nodes in the specified graph.

    It will connect the 'nodes_from' to the 'nodes_to' if an edge currently
    does *not* exist (if it does already exist then the edges attributes
    are just updated instead). When an edge is created the provided edge
    attributes dictionary will be applied to the new edge between these two
    nodes.
    """
    for u in nodes_from:
        for v in nodes_to:
            if not graph.has_edge(u, v):
                if attr_dict:
                    graph.add_edge(u, v, attr_dict=attr_dict.copy())
                else:
                    graph.add_edge(u, v)
            elif attr_dict:
                graph.add_edge(u, v, attr_dict=attr_dict.copy())