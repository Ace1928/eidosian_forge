import pytest
import networkx as nx
from networkx.utils import py_random_state
@py_random_state(1)
def _assign_random_weights(G, seed=None):
    """Assigns random weights to the edges of a graph.

    Parameters
    ----------

    G : NetworkX graph
        The original graph for which the spanner was constructed.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    for u, v in G.edges():
        G[u][v]['weight'] = seed.random()