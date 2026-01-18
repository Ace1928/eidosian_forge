from pyparsing import (
import pydot
def add_elements(g, toks, defaults_graph=None, defaults_node=None, defaults_edge=None):
    if defaults_graph is None:
        defaults_graph = {}
    if defaults_node is None:
        defaults_node = {}
    if defaults_edge is None:
        defaults_edge = {}
    for elm_idx, element in enumerate(toks):
        if isinstance(element, (pydot.Subgraph, pydot.Cluster)):
            add_defaults(element, defaults_graph)
            g.add_subgraph(element)
        elif isinstance(element, pydot.Node):
            add_defaults(element, defaults_node)
            g.add_node(element)
        elif isinstance(element, pydot.Edge):
            add_defaults(element, defaults_edge)
            g.add_edge(element)
        elif isinstance(element, ParseResults):
            for e in element:
                add_elements(g, [e], defaults_graph, defaults_node, defaults_edge)
        elif isinstance(element, DefaultStatement):
            if element.default_type == 'graph':
                default_graph_attrs = pydot.Node('graph', **element.attrs)
                g.add_node(default_graph_attrs)
            elif element.default_type == 'node':
                default_node_attrs = pydot.Node('node', **element.attrs)
                g.add_node(default_node_attrs)
            elif element.default_type == 'edge':
                default_edge_attrs = pydot.Node('edge', **element.attrs)
                g.add_node(default_edge_attrs)
                defaults_edge.update(element.attrs)
            else:
                raise ValueError('Unknown DefaultStatement: {s}'.format(s=element.default_type))
        elif isinstance(element, P_AttrList):
            g.obj_dict['attributes'].update(element.attrs)
        else:
            raise ValueError('Unknown element statement: {s}'.format(s=element))