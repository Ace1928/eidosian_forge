import pytest
import networkx as nx
from networkx.utils import py_random_state
def _test_spanner(G, spanner, stretch, weight=None):
    """Test whether a spanner is valid.

    This function tests whether the given spanner is a subgraph of the
    given graph G with the same node set. It also tests for all shortest
    paths whether they adhere to the given stretch.

    Parameters
    ----------
    G : NetworkX graph
        The original graph for which the spanner was constructed.

    spanner : NetworkX graph
        The spanner to be tested.

    stretch : float
        The proclaimed stretch of the spanner.

    weight : object
        The edge attribute to use as distance.
    """
    assert set(G.nodes()) == set(spanner.nodes())
    for u, v in spanner.edges():
        assert G.has_edge(u, v)
        if weight:
            assert spanner[u][v][weight] == G[u][v][weight]
    original_length = dict(nx.shortest_path_length(G, weight=weight))
    spanner_length = dict(nx.shortest_path_length(spanner, weight=weight))
    for u in G.nodes():
        for v in G.nodes():
            if u in original_length and v in original_length[u]:
                assert spanner_length[u][v] <= stretch * original_length[u][v]