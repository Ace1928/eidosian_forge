import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
class GraphBacking:
    """An ADT for a graph with hierarchical grouping so it is suited for
    representing regionalized flow graphs in SCFG.
    """
    _nodes: dict[str, GraphNode]
    _groups: GraphGroup
    _edges: set[GraphEdge]

    def __init__(self):
        self._nodes = {}
        self._groups = GraphGroup.make()
        self._edges = set()

    def add_node(self, name: str, node: GraphNode):
        """Add a graph node
        """
        assert name not in self._nodes
        self._nodes[name] = node
        group = self._groups
        for p in node.parent_regions:
            group = group.subgroups[p]
        group.nodes.add(name)

    def add_edge(self, src: str, dst: str, **kwargs):
        """Add a graph edge
        """
        self._edges.add(GraphEdge(src, dst, **kwargs))

    def verify(self):
        """Check graph structure.

        * check for missing nodes
        * check for missing ports
        """
        for edge in self._edges:
            if edge.src not in self._nodes:
                raise ValueError(f'missing node {edge.src!r}')
            if edge.dst not in self._nodes:
                raise ValueError(f'missing node {edge.dst!r}')
            if edge.src_port is not None:
                node = self._nodes[edge.src]
                if edge.src_port not in node.ports:
                    raise ValueError(f'missing port {edge.src_port!r} in node {edge.src!r}')
            if edge.dst_port is not None:
                node = self._nodes[edge.dst]
                if edge.dst_port not in node.ports:
                    raise ValueError(f'missing port {edge.dst_port!r} in node {edge.dst!r}')

    def render(self, renderer: 'AbstractRendererBackend'):
        """Render this graph using the given backend.
        """
        self._render_group(renderer, self._groups)
        for edge in self._edges:
            renderer.render_edge(edge)

    def _render_group(self, renderer, group: GraphGroup):
        """Recursively rendering the hierarchical groups
        """
        for k, subgroup in group.subgroups.items():
            with renderer.render_cluster(k) as subrenderer:
                self._render_group(subrenderer, subgroup)
        for k in group.nodes:
            node = self._nodes[k]
            renderer.render_node(k, node)