import networkx as nx
from collections import deque
def _to_sage_digraph(self):
    S = sage.graphs.graph.DiGraph(loops=True, multiedges=True)
    S.add_vertices(self.vertices)
    for e in self.edges:
        v, w = e
        S.add_edge(v, w, repr(e))
    return S