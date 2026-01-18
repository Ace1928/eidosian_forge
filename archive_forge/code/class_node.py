import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
class node:

    @staticmethod
    def _create(ckdtree_node=None):
        """Create either an inner or leaf node, wrapping a cKDTreeNode instance"""
        if ckdtree_node is None:
            return KDTree.node(ckdtree_node)
        elif ckdtree_node.split_dim == -1:
            return KDTree.leafnode(ckdtree_node)
        else:
            return KDTree.innernode(ckdtree_node)

    def __init__(self, ckdtree_node=None):
        if ckdtree_node is None:
            ckdtree_node = cKDTreeNode()
        self._node = ckdtree_node

    def __lt__(self, other):
        return id(self) < id(other)

    def __gt__(self, other):
        return id(self) > id(other)

    def __le__(self, other):
        return id(self) <= id(other)

    def __ge__(self, other):
        return id(self) >= id(other)

    def __eq__(self, other):
        return id(self) == id(other)