from .protocoltreenode import ProtocolTreeNode
import unittest, time
def _generateId(self, short=False):
    ProtocolEntity.__ID_GEN += 1
    return str(ProtocolEntity.__ID_GEN) if short else str(int(time.time())) + '-' + str(ProtocolEntity.__ID_GEN)