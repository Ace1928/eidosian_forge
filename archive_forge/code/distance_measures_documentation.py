import networkx as nx
from networkx.utils.decorators import py_random_state
Helper function for finding a lower bound on the diameter
        for directed Graphs.

        It implements 2-dSweep, the directed version of the 2-sweep algorithm.
        The algorithm follows the following steps.
        1. Select a source node $s$ at random.
        2. Perform a forward BFS from $s$ to select a node $a_1$ at the maximum
        distance from the source, and compute $LB_1$, the backward eccentricity of $a_1$.
        3. Perform a backward BFS from $s$ to select a node $a_2$ at the maximum
        distance from the source, and compute $LB_2$, the forward eccentricity of $a_2$.
        4. Return the maximum between $LB_1$ and $LB_2$.

        ``G`` is a NetworkX directed graph.

    .. note::

        ``seed`` is a random.Random or numpy.random.RandomState instance
    