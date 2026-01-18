import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def _get_leaves_color_list(R):
    leaves_color_list = [None] * len(R['leaves'])
    for link_x, link_y, link_color in zip(R['icoord'], R['dcoord'], R['color_list']):
        for xi, yi in zip(link_x, link_y):
            if yi == 0.0 and (xi % 5 == 0 and xi % 2 == 1):
                leaf_index = (int(xi) - 5) // 10
                leaves_color_list[leaf_index] = link_color
    return leaves_color_list