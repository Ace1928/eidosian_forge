import networkx as nx
import numpy as np
from scipy.sparse import linalg
from . import _ncut, _ncut_cy
Perform Normalized Graph cut on the Region Adjacency Graph.

    Recursively partition the graph into 2, until further subdivision
    yields a cut greater than `thresh` or such a cut cannot be computed.
    For such a subgraph, indices to labels of all its nodes map to a single
    unique value.

    Parameters
    ----------
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. A subgraph won't be further subdivided if the
        value of the N-cut exceeds `thresh`.
    num_cuts : int
        The number or N-cuts to perform before determining the optimal one.
    random_generator : `numpy.random.Generator`
        Provides initial values for eigenvalue solver.
    