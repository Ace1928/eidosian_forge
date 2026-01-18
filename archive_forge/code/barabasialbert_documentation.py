import numpy as np
from scipy import sparse
from . import Graph  # prevent circular import in Python < 3.5
Barabasi-Albert preferential attachment.

    The Barabasi-Albert graph is constructed by connecting nodes in two steps.
    First, m0 nodes are created. Then, nodes are added one by one.

    By lack of clarity, we take the liberty to create it as follows:

        1. the m0 initial nodes are disconnected,
        2. each node is connected to m of the older nodes with a probability
           distribution depending of the node-degrees of the other nodes,
           :math:`p_n(i) = \frac{1 + k_i}{\sum_j{1 + k_j}}`.

    Parameters
    ----------
    N : int
        Number of nodes (default is 1000)
    m0 : int
        Number of initial nodes (default is 1)
    m : int
        Number of connections at each step (default is 1)
        m can never be larger than m0.
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.BarabasiAlbert(N=150, seed=42)
    >>> G.set_coordinates(kind='spring', seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    