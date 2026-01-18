import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
Design an exponential window filter.

    Parameters
    ----------
    G : graph
    bmax : float
        Maximum relative band (default = 0.2)
    a : int
        Slope parameter (default = 1)

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Expwin(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    