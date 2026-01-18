import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require
def _add_edge_filter(values, graph):
    """Create edge in `graph` between central element of `values` and the rest.

    Add an edge between the middle element in `values` and
    all other elements of `values` into `graph`.  ``values[len(values) // 2]``
    is expected to be the central value of the footprint used.

    Parameters
    ----------
    values : array
        The array to process.
    graph : RAG
        The graph to add edges in.

    Returns
    -------
    0 : float
        Always returns 0. The return value is required so that `generic_filter`
        can put it in the output array, but it is ignored by this filter.
    """
    values = values.astype(int)
    center = values[len(values) // 2]
    for value in values:
        if value != center and (not graph.has_edge(center, value)):
            graph.add_edge(center, value)
    return 0.0