from ..Node import Node
from .common import CtrlNode
class UniOpNode(Node):
    """Generic node for performing any operation like Out = In.fn()"""

    def __init__(self, name, fn):
        self.fn = fn
        Node.__init__(self, name, terminals={'In': {'io': 'in'}, 'Out': {'io': 'out', 'bypass': 'In'}})

    def process(self, **args):
        return {'Out': getattr(args['In'], self.fn)()}