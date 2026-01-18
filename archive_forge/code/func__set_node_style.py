import logging
from .nesting import NestedState
from .diagrams_base import BaseGraph
def _set_node_style(self, state, style):
    try:
        node = self.fsm_graph.get_node(state)
        style_attr = self.fsm_graph.style_attributes.get('node', {}).get(style)
        node.attr.update(style_attr)
    except KeyError:
        subgraph = _get_subgraph(self.fsm_graph, 'cluster_' + state)
        style_attr = self.fsm_graph.style_attributes.get('graph', {}).get(style)
        subgraph.graph_attr.update(style_attr)