import networkx as nx
from collections import deque
def _to_sage(self, loops=True, multiedges=True):
    S = sage.graphs.graph.Graph(loops=loops, multiedges=multiedges)
    S.add_vertices(self.vertices)
    for e in self.edges:
        v, w = e
        S.add_edge(v, w, repr(e))
    return S