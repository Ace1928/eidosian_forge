from ..Node import Node
from .common import CtrlNode
class MultiplyNode(BinOpNode):
    """Returns A * B. Does not check input types."""
    nodeName = 'Multiply'

    def __init__(self, name):
        BinOpNode.__init__(self, name, '__mul__')