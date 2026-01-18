from __future__ import annotations
import numpy as np
import param
import scipy.sparse
class circular_layout(LayoutAlgorithm):
    """
    Assign coordinates to the nodes along a circle.

    The points on the circle can be spaced either uniformly or randomly.

    Accepts an edges argument for consistency with other layout algorithms,
    but ignores it.
    """
    uniform = param.Boolean(True, doc='\n        Whether to distribute nodes evenly on circle')

    def __call__(self, nodes, edges=None, **params):
        p = param.ParamOverrides(self, params)
        np.random.seed(p.seed)
        r = 0.5
        x0, y0 = (0.5, 0.5)
        circumference = 2 * np.pi
        df = nodes.copy()
        if p.uniform:
            thetas = np.arange(circumference, step=circumference / len(df))
        else:
            thetas = np.asarray(np.random.random((len(df),))) * circumference
        df[p.x] = x0 + r * np.cos(thetas)
        df[p.y] = y0 + r * np.sin(thetas)
        return df