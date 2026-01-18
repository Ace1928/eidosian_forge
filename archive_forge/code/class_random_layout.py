from __future__ import annotations
import numpy as np
import param
import scipy.sparse
class random_layout(LayoutAlgorithm):
    """
    Assign coordinates to the nodes randomly.

    Accepts an edges argument for consistency with other layout algorithms,
    but ignores it.
    """

    def __call__(self, nodes, edges=None, **params):
        p = param.ParamOverrides(self, params)
        np.random.seed(p.seed)
        df = nodes.copy()
        points = np.asarray(np.random.random((len(df), 2)))
        df[p.x] = points[:, 0]
        df[p.y] = points[:, 1]
        return df