import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
Returns the LFR benchmark graph.

    This algorithm proceeds as follows:

    1) Find a degree sequence with a power law distribution, and minimum
       value ``min_degree``, which has approximate average degree
       ``average_degree``. This is accomplished by either

       a) specifying ``min_degree`` and not ``average_degree``,
       b) specifying ``average_degree`` and not ``min_degree``, in which
          case a suitable minimum degree will be found.

       ``max_degree`` can also be specified, otherwise it will be set to
       ``n``. Each node *u* will have $\mu \mathrm{deg}(u)$ edges
       joining it to nodes in communities other than its own and $(1 -
       \mu) \mathrm{deg}(u)$ edges joining it to nodes in its own
       community.
    2) Generate community sizes according to a power law distribution
       with exponent ``tau2``. If ``min_community`` and
       ``max_community`` are not specified they will be selected to be
       ``min_degree`` and ``max_degree``, respectively.  Community sizes
       are generated until the sum of their sizes equals ``n``.
    3) Each node will be randomly assigned a community with the
       condition that the community is large enough for the node's
       intra-community degree, $(1 - \mu) \mathrm{deg}(u)$ as
       described in step 2. If a community grows too large, a random node
       will be selected for reassignment to a new community, until all
       nodes have been assigned a community.
    4) Each node *u* then adds $(1 - \mu) \mathrm{deg}(u)$
       intra-community edges and $\mu \mathrm{deg}(u)$ inter-community
       edges.

    Parameters
    ----------
    n : int
        Number of nodes in the created graph.

    tau1 : float
        Power law exponent for the degree distribution of the created
        graph. This value must be strictly greater than one.

    tau2 : float
        Power law exponent for the community size distribution in the
        created graph. This value must be strictly greater than one.

    mu : float
        Fraction of inter-community edges incident to each node. This
        value must be in the interval [0, 1].

    average_degree : float
        Desired average degree of nodes in the created graph. This value
        must be in the interval [0, *n*]. Exactly one of this and
        ``min_degree`` must be specified, otherwise a
        :exc:`NetworkXError` is raised.

    min_degree : int
        Minimum degree of nodes in the created graph. This value must be
        in the interval [0, *n*]. Exactly one of this and
        ``average_degree`` must be specified, otherwise a
        :exc:`NetworkXError` is raised.

    max_degree : int
        Maximum degree of nodes in the created graph. If not specified,
        this is set to ``n``, the total number of nodes in the graph.

    min_community : int
        Minimum size of communities in the graph. If not specified, this
        is set to ``min_degree``.

    max_community : int
        Maximum size of communities in the graph. If not specified, this
        is set to ``n``, the total number of nodes in the graph.

    tol : float
        Tolerance when comparing floats, specifically when comparing
        average degree values.

    max_iters : int
        Maximum number of iterations to try to create the community sizes,
        degree distribution, and community affiliations.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : NetworkX graph
        The LFR benchmark graph generated according to the specified
        parameters.

        Each node in the graph has a node attribute ``'community'`` that
        stores the community (that is, the set of nodes) that includes
        it.

    Raises
    ------
    NetworkXError
        If any of the parameters do not meet their upper and lower bounds:

        - ``tau1`` and ``tau2`` must be strictly greater than 1.
        - ``mu`` must be in [0, 1].
        - ``max_degree`` must be in {1, ..., *n*}.
        - ``min_community`` and ``max_community`` must be in {0, ...,
          *n*}.

        If not exactly one of ``average_degree`` and ``min_degree`` is
        specified.

        If ``min_degree`` is not specified and a suitable ``min_degree``
        cannot be found.

    ExceededMaxIterations
        If a valid degree sequence cannot be created within
        ``max_iters`` number of iterations.

        If a valid set of community sizes cannot be created within
        ``max_iters`` number of iterations.

        If a valid community assignment cannot be created within ``10 *
        n * max_iters`` number of iterations.

    Examples
    --------
    Basic usage::

        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> n = 250
        >>> tau1 = 3
        >>> tau2 = 1.5
        >>> mu = 0.1
        >>> G = LFR_benchmark_graph(
        ...     n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10
        ... )

    Continuing the example above, you can get the communities from the
    node attributes of the graph::

        >>> communities = {frozenset(G.nodes[v]["community"]) for v in G}

    Notes
    -----
    This algorithm differs slightly from the original way it was
    presented in [1].

    1) Rather than connecting the graph via a configuration model then
       rewiring to match the intra-community and inter-community
       degrees, we do this wiring explicitly at the end, which should be
       equivalent.
    2) The code posted on the author's website [2] calculates the random
       power law distributed variables and their average using
       continuous approximations, whereas we use the discrete
       distributions here as both degree and community size are
       discrete.

    Though the authors describe the algorithm as quite robust, testing
    during development indicates that a somewhat narrower parameter set
    is likely to successfully produce a graph. Some suggestions have
    been provided in the event of exceptions.

    References
    ----------
    .. [1] "Benchmark graphs for testing community detection algorithms",
           Andrea Lancichinetti, Santo Fortunato, and Filippo Radicchi,
           Phys. Rev. E 78, 046110 2008
    .. [2] https://www.santofortunato.net/resources

    