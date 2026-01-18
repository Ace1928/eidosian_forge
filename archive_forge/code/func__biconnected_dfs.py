from itertools import chain
import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('directed')
def _biconnected_dfs(G, components=True):
    visited = set()
    for start in G:
        if start in visited:
            continue
        discovery = {start: 0}
        low = {start: 0}
        root_children = 0
        visited.add(start)
        edge_stack = []
        stack = [(start, start, iter(G[start]))]
        edge_index = {}
        while stack:
            grandparent, parent, children = stack[-1]
            try:
                child = next(children)
                if grandparent == child:
                    continue
                if child in visited:
                    if discovery[child] <= discovery[parent]:
                        low[parent] = min(low[parent], discovery[child])
                        if components:
                            edge_index[parent, child] = len(edge_stack)
                            edge_stack.append((parent, child))
                else:
                    low[child] = discovery[child] = len(discovery)
                    visited.add(child)
                    stack.append((parent, child, iter(G[child])))
                    if components:
                        edge_index[parent, child] = len(edge_stack)
                        edge_stack.append((parent, child))
            except StopIteration:
                stack.pop()
                if len(stack) > 1:
                    if low[parent] >= discovery[grandparent]:
                        if components:
                            ind = edge_index[grandparent, parent]
                            yield edge_stack[ind:]
                            del edge_stack[ind:]
                        else:
                            yield grandparent
                    low[grandparent] = min(low[parent], low[grandparent])
                elif stack:
                    root_children += 1
                    if components:
                        ind = edge_index[grandparent, parent]
                        yield edge_stack[ind:]
                        del edge_stack[ind:]
        if not components:
            if root_children > 1:
                yield start