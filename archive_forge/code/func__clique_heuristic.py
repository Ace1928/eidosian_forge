import networkx as nx
from networkx.algorithms.approximation import ramsey
from networkx.utils import not_implemented_for
def _clique_heuristic(G, U, size, best_size):
    if not U:
        return max(best_size, size)
    u = max(U, key=degrees)
    U.remove(u)
    N_prime = {v for v in G[u] if degrees[v] >= best_size}
    return _clique_heuristic(G, U & N_prime, size + 1, best_size)