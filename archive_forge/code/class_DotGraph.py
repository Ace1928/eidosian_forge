import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
class DotGraph(object):
    """Class representing a DOT graph"""

    def __init__(self, name='G', strict=True, directed=False, **kwds):
        self._nodes = OrderedDict()
        self._allnodes = {}
        self._alledges = {}
        self._allgraphs = []
        self._edges = {}
        self.strict = strict
        self.directed = directed
        self.subgraphs = []
        self.name = name
        self.padding = '    '
        self.seq = 0
        self.allitems = []
        self.attr = {}
        self.strict = strict
        self.level = 0
        self.parent = None
        self.root = self
        self.adj = {}
        self.default_node_attr = {}
        self.default_edge_attr = {}
        self.default_graph_attr = {}
        self.attr.update(kwds)
        self._allgraphs.append(self)
        pass

    def __len__(self):
        return len(self._nodes) + sum((len(s) for s in self.subgraphs))

    def __getattr__(self, name):
        try:
            return self.attr[name]
        except KeyError:
            raise AttributeError

    def get_name(self):
        if self.name.strip():
            return quote_if_necessary(self.name)
        else:
            return ''

    def add_node(self, node, **kwds):
        if not isinstance(node, DotNode):
            node = DotNode(str(node), **kwds)
        n = node.name
        if n in self._allnodes:
            self._allnodes[n].attr.update(kwds)
        else:
            node.attr.update(self.default_node_attr)
            node.attr.update(kwds)
            self._allnodes[n] = node
        if n not in self._nodes:
            self._nodes[n] = node
        return node

    def add_edge(self, src, dst, srcport='', dstport='', **kwds):
        u = self.add_node(src)
        v = self.add_node(dst)
        edge = DotEdge(u, v, self.directed, srcport, dstport, **self.default_edge_attr)
        edge.attr.update(kwds)
        edgekey = (u.name, v.name)
        if edgekey in self._alledges:
            edgs = self._alledges[edgekey]
            if not self.strict:
                if edgekey in self._edges:
                    self._edges[edgekey].append(edge)
                edgs.append(edge)
        else:
            self._alledges[edgekey] = [edge]
            self._edges[edgekey] = [edge]
        return edge

    def add_special_edge(self, src, dst, srcport='', dstport='', **kwds):
        src_is_graph = isinstance(src, DotSubGraph)
        dst_is_graph = isinstance(dst, DotSubGraph)
        edges = []
        if src_is_graph:
            src_nodes = src.get_all_nodes()
        else:
            src_nodes = [src]
        if dst_is_graph:
            dst_nodes = dst.get_all_nodes()
        else:
            dst_nodes = [dst]
        for src_node in src_nodes:
            for dst_node in dst_nodes:
                edge = self.add_edge(src_node, dst_node, srcport, dstport, **kwds)
                edges.append(edge)
        return edges

    def add_default_node_attr(self, **kwds):
        self.default_node_attr.update(kwds)

    def add_default_edge_attr(self, **kwds):
        self.default_edge_attr.update(kwds)

    def add_default_graph_attr(self, **kwds):
        self.default_graph_attr.update(kwds)

    def delete_node(self, node):
        if isinstance(node, DotNode):
            name = node.name
        else:
            name = node
        try:
            del self._nodes[name]
            del self._allnodes[name]
        except:
            raise DotParsingException('Node %s does not exists' % name)

    def get_node(self, nodename):
        """Return node with name=nodename

        Returns None if node does not exists.
        """
        return self._allnodes.get(nodename, None)

    def add_subgraph(self, subgraph, **kwds):
        if isinstance(subgraph, DotSubGraph):
            subgraphcls = subgraph
        else:
            subgraphcls = DotSubGraph(subgraph, self.strict, self.directed, **kwds)
        subgraphcls._allnodes = self._allnodes
        subgraphcls._alledges = self._alledges
        subgraphcls._allgraphs = self._allgraphs
        subgraphcls.parent = self
        subgraphcls.root = self.root
        subgraphcls.level = self.level + 1
        subgraphcls.add_default_node_attr(**self.default_node_attr)
        subgraphcls.add_default_edge_attr(**self.default_edge_attr)
        subgraphcls.add_default_graph_attr(**self.attr)
        subgraphcls.attr.update(self.default_graph_attr)
        subgraphcls.padding += self.padding
        self.subgraphs.append(subgraphcls)
        self._allgraphs.append(subgraphcls)
        return subgraphcls

    def get_subgraphs(self):
        return self.subgraphs

    def get_edges(self):
        return self._edges

    def get_all_nodes(self):
        nodes = []
        for subgraph in self.get_subgraphs():
            nodes.extend(subgraph.get_all_nodes())
        nodes.extend(self._nodes)
        return nodes

    def set_attr(self, **kwds):
        """Set graph attributes"""
        self.attr.update(kwds)
    nodes = property(lambda self: self._nodes.values())
    allnodes = property(lambda self: self._allnodes.values())
    allgraphs = property(lambda self: self._allgraphs.__iter__())
    alledges = property(lambda self: flatten(self._alledges.values()))
    edges = property(get_edges)

    def __str__(self):
        s = ''
        padding = self.padding
        if len(self.allitems) > 0:
            grstr = ''.join(['%s%s' % (padding, n) for n in map(str, flatten(self.allitems))])
            attrstr = ','.join(['%s=%s' % (quote_if_necessary(key), quote_if_necessary(val)) for key, val in self.attr.items()])
            if attrstr:
                attrstr = '%sgraph [%s];' % (padding, attrstr)
            if not isinstance(self, DotSubGraph):
                s = ''
                if self.strict:
                    s += 'strict '
                if self.directed:
                    s += 'digraph'
                else:
                    s += 'graph'
                return '%s %s{\n%s\n%s\n}' % (s, self.get_name(), grstr, attrstr)
            else:
                return '%s %s{\n%s\n%s\n%s}' % ('subgraph', self.get_name(), grstr, attrstr, padding)
        subgraphstr = '\n'.join(['%s%s' % (padding, n) for n in map(str, self.subgraphs)])
        nodestr = ''.join(['%s%s' % (padding, n) for n in map(str, self._nodes.values())])
        edgestr = ''.join(['%s%s' % (padding, n) for n in map(str, flatten(self.edges.values()))])
        attrstr = ','.join(['%s=%s' % (quote_if_necessary(key), quote_if_necessary(val)) for key, val in self.attr.items()])
        if attrstr:
            attrstr = '%sgraph [%s];' % (padding, attrstr)
        if not isinstance(self, DotSubGraph):
            s = ''
            if self.strict:
                s += 'strict '
            if self.directed:
                s += 'digraph'
            else:
                s += 'graph'
            return '%s %s{\n%s\n%s\n%s\n%s\n}' % (s, self.get_name(), subgraphstr, attrstr, nodestr, edgestr)
        else:
            return '%s %s{\n%s\n%s\n%s\n%s\n%s}' % ('subgraph', self.get_name(), subgraphstr, attrstr, nodestr, edgestr, padding)