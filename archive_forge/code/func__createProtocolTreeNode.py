from .protocoltreenode import ProtocolTreeNode
import unittest, time
def _createProtocolTreeNode(self, attributes, children=None, data=None):
    return ProtocolTreeNode(self.getTag(), attributes, children, data)