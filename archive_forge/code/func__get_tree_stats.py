import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
def _get_tree_stats(self):
    """
        Return a python list containing the statistics about the node tree:
            0: number of nodes (tree size)
            1: number of unique nodes
            2: number of trapezoids (tree leaf nodes)
            3: number of unique trapezoids
            4: maximum parent count (max number of times a node is repeated in
                   tree)
            5: maximum depth of tree (one more than the maximum number of
                   comparisons needed to search through the tree)
            6: mean of all trapezoid depths (one more than the average number
                   of comparisons needed to search through the tree)
        """
    return self._cpp_trifinder.get_tree_stats()