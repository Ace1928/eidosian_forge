from warnings import warn
import networkx as nx
def _pagerank_python(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None):
    if len(G) == 0:
        return {}
    D = G.to_directed()
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        s = sum(nstart.values())
        x = {k: v / s for k, v in nstart.items()}
    if personalization is None:
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = sum(personalization.values())
        p = {k: v / s for k, v in personalization.items()}
    if dangling is None:
        dangling_weights = p
    else:
        s = sum(dangling.values())
        dangling_weights = {k: v / s for k, v in dangling.items()}
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum((xlast[n] for n in dangling_nodes))
        for n in x:
            for _, nbr, wt in W.edges(n, data=weight):
                x[nbr] += alpha * xlast[n] * wt
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        err = sum((abs(x[n] - xlast[n]) for n in x))
        if err < N * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)