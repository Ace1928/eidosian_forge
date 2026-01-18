from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
import sage.graphs.graph as graph
from sage.rings.rational_field import QQ
def _recursive_spanning_trees(G, forest):
    """
        Returns all the spanning trees of G containing forest
        """
    if not G.is_connected():
        return []
    if G.size() == forest.size():
        return [forest.copy()]
    else:
        for e in G.edges(sort=True, key=edge_index):
            if not forest.has_edge(e):
                break
        G.delete_edge(e)
        trees = _recursive_spanning_trees(G, forest)
        G.add_edge(e)
        c1 = forest.connected_component_containing_vertex(e[0])
        c2 = forest.connected_component_containing_vertex(e[1])
        G.delete_edge(e)
        B = G.edge_boundary(c1, c2, sort=False)
        G.add_edge(e)
        forest.add_edge(e)
        G.delete_edges(B)
        trees.extend(_recursive_spanning_trees(G, forest))
        G.add_edges(B)
        forest.delete_edge(e)
        return trees