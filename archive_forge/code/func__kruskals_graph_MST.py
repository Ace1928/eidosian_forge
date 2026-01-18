import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def _kruskals_graph_MST(graph):
    """Find the minimal spanning tree in graph using Kruskals algorithm.

    See http://en.wikipedia.org/wiki/Kruskal%27s_algorithm.
    :param graph: A graph in {from:{to:value}} form. Every node present in
        graph must be in the outer dict (because graph is not a directed graph.
    :return: A graph with all nodes and those vertices that are part of the MST
        for graph. If graph is not connected, then the result will also be a
        forest.
    """
    forest = {}
    for node in graph:
        forest[node] = {node: {}}
    graphs = len(forest)
    edges = set()
    for from_node, to_nodes in graph.items():
        for to_node, value in to_nodes.items():
            edge = (value,) + tuple(sorted([from_node, to_node]))
            edges.add(edge)
    edges = list(edges)
    heapq.heapify(edges)
    while edges and graphs > 1:
        edge = heapq.heappop(edges)
        g1 = forest[edge[1]]
        g2 = forest[edge[2]]
        if g1 is g2:
            continue
        graphs -= 1
        for from_node, to_nodes in g2.items():
            forest[from_node] = g1
            g1.setdefault(from_node, {}).update(to_nodes)
        g1[edge[1]][edge[2]] = edge[0]
        g1[edge[2]][edge[1]] = edge[0]
    _, result = forest.popitem()
    for _, g2 in forest.items():
        if g2 is result:
            continue
        for from_node, to_nodes in g2.items():
            result.setdefault(from_node, {}).update(to_nodes)
    return result