import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
class leafnode(node):

    @property
    def idx(self):
        return self._node.indices

    @property
    def children(self):
        return self._node.children