import numpy as np
from . import Graph  # prevent circular import in Python < 3.5
Fully connected graph.

    All weights are set to 1. There is no self-connections.

    Parameters
    ----------
    N : int
        Number of vertices (default = 10)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.FullConnected(N=20)
    >>> G.set_coordinates(kind='spring', seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=5)
    >>> G.plot(ax=axes[1])

    