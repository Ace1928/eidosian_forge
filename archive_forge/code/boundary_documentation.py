from itertools import chain
import networkx as nx
Returns the node boundary of `nbunch1`.

    The *node boundary* of a set *S* with respect to a set *T* is the
    set of nodes *v* in *T* such that for some *u* in *S*, there is an
    edge joining *u* to *v*. If *T* is not specified, it is assumed to
    be the set of all nodes not in *S*.

    Parameters
    ----------
    G : NetworkX graph

    nbunch1 : iterable
        Iterable of nodes in the graph representing the set of nodes
        whose node boundary will be returned. (This is the set *S* from
        the definition above.)

    nbunch2 : iterable
        Iterable of nodes representing the target (or "exterior") set of
        nodes. (This is the set *T* from the definition above.) If not
        specified, this is assumed to be the set of all nodes in `G`
        not in `nbunch1`.

    Returns
    -------
    set
        The node boundary of `nbunch1` with respect to `nbunch2`.

    Examples
    --------
    >>> G = nx.wheel_graph(6)

    When nbunch2=None:

    >>> list(nx.node_boundary(G, (3, 4)))
    [0, 2, 5]

    When nbunch2 is given:

    >>> list(nx.node_boundary(G, (3, 4), (0, 1, 5)))
    [0, 5]

    Notes
    -----
    Any element of `nbunch` that is not in the graph `G` will be
    ignored.

    `nbunch1` and `nbunch2` are usually meant to be disjoint, but in
    the interest of speed and generality, that is not required here.

    