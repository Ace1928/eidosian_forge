import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def DAG_from_direction(self, kind):
    H = Digraph(pairs=[e for e in self.edges if e.kind == kind], singles=self.vertices)
    maximal_chains = H.weak_components()
    vertex_to_chain = element_map(maximal_chains)
    D = Digraph(singles=maximal_chains)
    for e in [e for e in self.edges if e.kind != kind]:
        d = D.add_edge(vertex_to_chain[e.tail], vertex_to_chain[e.head])
        d.dummy = e in self.dummy
    for u, v in self.saturation_edges(False):
        d = D.add_edge(vertex_to_chain[u], vertex_to_chain[v])
        d.dummy = True
    for u, v in self.saturation_edges(True):
        if kind == 'vertical':
            u, v = (v, u)
        d = D.add_edge(vertex_to_chain[u], vertex_to_chain[v])
        d.dummy = True
    D.vertex_to_chain = vertex_to_chain
    return D