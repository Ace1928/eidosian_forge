import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
class innernode(node):

    def __init__(self, ckdtreenode):
        assert isinstance(ckdtreenode, cKDTreeNode)
        super().__init__(ckdtreenode)
        self.less = KDTree.node._create(ckdtreenode.lesser)
        self.greater = KDTree.node._create(ckdtreenode.greater)

    @property
    def split_dim(self):
        return self._node.split_dim

    @property
    def split(self):
        return self._node.split

    @property
    def children(self):
        return self._node.children