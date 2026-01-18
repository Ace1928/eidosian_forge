from __future__ import annotations
import numpy as np
import param
import scipy.sparse
def _merge_points_with_nodes(nodes, points, params):
    n = nodes.copy()
    n[params.x] = points[:, 0]
    n[params.y] = points[:, 1]
    return n