from ..Node import Node
from .common import CtrlNode
class DivideNode(BinOpNode):
    """Returns A / B. Does not check input types."""
    nodeName = 'Divide'

    def __init__(self, name):
        BinOpNode.__init__(self, name, ('__truediv__', '__div__'))