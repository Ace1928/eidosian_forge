from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import not_implemented_for, pairwise
def _all_simple_paths_multigraph(G, source, targets, cutoff):
    visited = {source: True}
    stack = [(v for u, v in G.edges(source))]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child in targets:
                yield (list(visited) + [child])
            visited[child] = True
            if targets - set(visited.keys()):
                stack.append((v for u, v in G.edges(child)))
            else:
                visited.popitem()
        else:
            for target in targets - set(visited.keys()):
                count = ([child] + list(children)).count(target)
                for i in range(count):
                    yield (list(visited) + [target])
            stack.pop()
            visited.popitem()