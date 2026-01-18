import numpy as np
from scipy import sparse
from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5
Airfoil graph.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Airfoil()
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.5)
    >>> G.plot(show_edges=True, ax=axes[1])

    