from itertools import combinations
import networkx as nx
from ..utils import arbitrary_element, not_implemented_for
def _multigraph_eulerian_circuit(G, source):
    if G.is_directed():
        degree = G.out_degree
        edges = G.out_edges
    else:
        degree = G.degree
        edges = G.edges
    vertex_stack = [(source, None)]
    last_vertex = None
    last_key = None
    while vertex_stack:
        current_vertex, current_key = vertex_stack[-1]
        if degree(current_vertex) == 0:
            if last_vertex is not None:
                yield (last_vertex, current_vertex, last_key)
            last_vertex, last_key = (current_vertex, current_key)
            vertex_stack.pop()
        else:
            triple = arbitrary_element(edges(current_vertex, keys=True))
            _, next_vertex, next_key = triple
            vertex_stack.append((next_vertex, next_key))
            G.remove_edge(current_vertex, next_vertex, next_key)