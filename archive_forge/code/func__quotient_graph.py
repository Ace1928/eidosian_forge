from itertools import chain, combinations, permutations, product
import networkx as nx
from networkx import density
from networkx.exception import NetworkXException
from networkx.utils import arbitrary_element
def _quotient_graph(G, partition, edge_relation, node_data, edge_data, weight, relabel, create_using):
    """Construct the quotient graph assuming input has been checked"""
    if create_using is None:
        H = G.__class__()
    else:
        H = nx.empty_graph(0, create_using)
    if node_data is None:

        def node_data(b):
            S = G.subgraph(b)
            return {'graph': S, 'nnodes': len(S), 'nedges': S.number_of_edges(), 'density': density(S)}
    partition = [frozenset(b) for b in partition]
    H.add_nodes_from(((b, node_data(b)) for b in partition))
    if edge_relation is None:

        def edge_relation(b, c):
            return any((v in G[u] for u, v in product(b, c)))
    if edge_data is None:

        def edge_data(b, c):
            edgedata = (d for u, v, d in G.edges(b | c, data=True) if u in b and v in c or (u in c and v in b))
            return {'weight': sum((d.get(weight, 1) for d in edgedata))}
    block_pairs = permutations(H, 2) if H.is_directed() else combinations(H, 2)
    if H.is_multigraph():
        edges = chaini((((b, c, G.get_edge_data(u, v, default={})) for u, v in product(b, c) if v in G[u]) for b, c in block_pairs if edge_relation(b, c)))
    else:
        edges = ((b, c, edge_data(b, c)) for b, c in block_pairs if edge_relation(b, c))
    H.add_edges_from(edges)
    if relabel:
        labels = {b: i for i, b in enumerate(partition)}
        H = nx.relabel_nodes(H, labels)
    return H