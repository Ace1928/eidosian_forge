import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def _makeOrder(self, partition):
    """Return a order for the resource sets in partition."""
    root = frozenset(['root'])
    partition.add(root)
    partition.discard(frozenset())
    digraph = self._getGraph(partition)
    primes = {}
    prime = frozenset(['prime'])
    for node in digraph:
        primes[node] = node.union(prime)
    graph = _digraph_to_graph(digraph, primes)
    mst = _kruskals_graph_MST(graph)
    node = root
    cycle = [node]
    steps = 2 * (len(mst) - 1)
    for step in range(steps):
        found = False
        outgoing = None
        for outgoing in mst[node]:
            if node in mst[outgoing]:
                del mst[node][outgoing]
                node = outgoing
                cycle.append(node)
                found = True
                break
        if not found:
            del mst[node][outgoing]
            node = outgoing
            cycle.append(node)
    visited = set()
    order = []
    for node in cycle:
        if node in visited:
            continue
        if node in primes:
            order.append(node)
        visited.add(node)
    assert order[0] == root
    return order[1:]