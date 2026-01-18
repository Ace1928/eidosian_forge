from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import not_implemented_for, pairwise
def _all_simple_edge_paths_multigraph(G, source, targets, cutoff):
    if not cutoff or cutoff < 1:
        return []
    visited = [source]
    stack = [iter(G.edges(source, keys=True))]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.pop()
        elif len(visited) < cutoff:
            if child[1] in targets:
                yield (visited[1:] + [child])
            elif child[1] not in [v[0] for v in visited[1:]]:
                visited.append(child)
                stack.append(iter(G.edges(child[1], keys=True)))
        else:
            for u, v, k in [child] + list(children):
                if v in targets:
                    yield (visited[1:] + [(u, v, k)])
            stack.pop()
            visited.pop()