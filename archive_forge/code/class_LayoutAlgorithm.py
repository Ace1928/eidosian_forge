from __future__ import annotations
import numpy as np
import param
import scipy.sparse
class LayoutAlgorithm(param.ParameterizedFunction):
    """
    Baseclass for all graph layout algorithms.
    """
    __abstract = True
    seed = param.Integer(default=None, bounds=(0, 2 ** 32 - 1), doc='\n        Random seed used to initialize the pseudo-random number\n        generator.')
    x = param.String(default='x', doc="\n        Column name for each node's x coordinate.")
    y = param.String(default='y', doc="\n        Column name for each node's y coordinate.")
    source = param.String(default='source', doc="\n        Column name for each edge's source.")
    target = param.String(default='target', doc="\n        Column name for each edge's target.")
    weight = param.String(default=None, allow_None=True, doc='\n        Column name for each edge weight. If None, weights are ignored.')
    id = param.String(default=None, allow_None=True, doc='\n        Column name for a unique identifier for the node.  If None, the\n        dataframe index is used.')

    def __call__(self, nodes, edges, **params):
        """
        This method takes two dataframes representing a graph's nodes
        and edges respectively. For the nodes dataframe, the only
        column accessed is the specified `id` value (or the index if
        no 'id'). For the edges dataframe, the columns are `id`,
        `source`, `target`, and (optionally) `weight`.

        Each layout algorithm will use the two dataframes as appropriate to
        assign positions to the nodes. Upon generating positions, this
        method will return a copy of the original nodes dataframe with
        two additional columns for the x and y coordinates.
        """
        return NotImplementedError