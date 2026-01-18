from typing import Type, AbstractSet
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial
from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree
class ForestToPyDotVisitor(ForestVisitor):
    """
    A Forest visitor which writes the SPPF to a PNG.

    The SPPF can get really large, really quickly because
    of the amount of meta-data it stores, so this is probably
    only useful for trivial trees and learning how the SPPF
    is structured.
    """

    def __init__(self, rankdir='TB'):
        super(ForestToPyDotVisitor, self).__init__(single_visit=True)
        self.pydot = import_module('pydot')
        self.graph = self.pydot.Dot(graph_type='digraph', rankdir=rankdir)

    def visit(self, root, filename):
        super(ForestToPyDotVisitor, self).visit(root)
        try:
            self.graph.write_png(filename)
        except FileNotFoundError as e:
            logger.error('Could not write png: ', e)

    def visit_token_node(self, node):
        graph_node_id = str(id(node))
        graph_node_label = '"{}"'.format(node.value.replace('"', '\\"'))
        graph_node_color = 8421504
        graph_node_style = '"filled,rounded"'
        graph_node_shape = 'diamond'
        graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor='#{:06x}'.format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
        self.graph.add_node(graph_node)

    def visit_packed_node_in(self, node):
        graph_node_id = str(id(node))
        graph_node_label = repr(node)
        graph_node_color = 8421504
        graph_node_style = 'filled'
        graph_node_shape = 'diamond'
        graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor='#{:06x}'.format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
        self.graph.add_node(graph_node)
        yield node.left
        yield node.right

    def visit_packed_node_out(self, node):
        graph_node_id = str(id(node))
        graph_node = self.graph.get_node(graph_node_id)[0]
        for child in [node.left, node.right]:
            if child is not None:
                child_graph_node_id = str(id(child.token if isinstance(child, TokenNode) else child))
                child_graph_node = self.graph.get_node(child_graph_node_id)[0]
                self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node))
            else:
                child_graph_node_id = str(randint(100000000000000000000000000000, 123456789012345678901234567890))
                child_graph_node_style = 'invis'
                child_graph_node = self.pydot.Node(child_graph_node_id, style=child_graph_node_style, label='None')
                child_edge_style = 'invis'
                self.graph.add_node(child_graph_node)
                self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node, style=child_edge_style))

    def visit_symbol_node_in(self, node):
        graph_node_id = str(id(node))
        graph_node_label = repr(node)
        graph_node_color = 8421504
        graph_node_style = '"filled"'
        if node.is_intermediate:
            graph_node_shape = 'ellipse'
        else:
            graph_node_shape = 'rectangle'
        graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor='#{:06x}'.format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
        self.graph.add_node(graph_node)
        return iter(node.children)

    def visit_symbol_node_out(self, node):
        graph_node_id = str(id(node))
        graph_node = self.graph.get_node(graph_node_id)[0]
        for child in node.children:
            child_graph_node_id = str(id(child))
            child_graph_node = self.graph.get_node(child_graph_node_id)[0]
            self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node))