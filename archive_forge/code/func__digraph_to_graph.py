import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def _digraph_to_graph(digraph, prime_node_mapping):
    """Convert digraph to a graph.

    :param digraph: A directed graph in the form
        {from:{to:value}}.
    :param prime_node_mapping: A mapping from every
        node in digraph to a new unique and not in digraph node.
    :return: A symmetric graph in the form {from:to:value}} created by
        creating edges in the result between every N to M-prime with the
        original N-M value and from every N to N-prime with a cost of 0.
        No other edges are created.
    """
    result = {}
    for from_node, from_prime_node in prime_node_mapping.items():
        result[from_node] = {from_prime_node: 0}
        result[from_prime_node] = {from_node: 0}
    for from_node, to_nodes in digraph.items():
        from_prime = prime_node_mapping[from_node]
        for to_node, value in to_nodes.items():
            to_prime = prime_node_mapping[to_node]
            result[from_prime][to_node] = value
            result[to_node][from_prime] = value
    return result