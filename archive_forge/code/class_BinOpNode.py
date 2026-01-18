from ..Node import Node
from .common import CtrlNode
class BinOpNode(CtrlNode):
    """Generic node for performing any operation like A.fn(B)"""
    _dtypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']
    uiTemplate = [('outputType', 'combo', {'values': ['no change', 'input A', 'input B'] + _dtypes, 'index': 0})]

    def __init__(self, name, fn):
        self.fn = fn
        CtrlNode.__init__(self, name, terminals={'A': {'io': 'in'}, 'B': {'io': 'in'}, 'Out': {'io': 'out', 'bypass': 'A'}})

    def process(self, **args):
        if isinstance(self.fn, tuple):
            for name in self.fn:
                try:
                    fn = getattr(args['A'], name)
                    break
                except AttributeError as e:
                    pass
            else:
                raise e
        else:
            fn = getattr(args['A'], self.fn)
        out = fn(args['B'])
        if out is NotImplemented:
            raise Exception('Operation %s not implemented between %s and %s' % (fn, str(type(args['A'])), str(type(args['B']))))
        typ = self.stateGroup.state()['outputType']
        if typ == 'no change':
            pass
        elif typ == 'input A':
            out = out.astype(args['A'].dtype)
        elif typ == 'input B':
            out = out.astype(args['B'].dtype)
        else:
            out = out.astype(typ)
        return {'Out': out}