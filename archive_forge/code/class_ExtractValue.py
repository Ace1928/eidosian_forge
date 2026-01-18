from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class ExtractValue(Instruction):

    def __init__(self, parent, agg, indices, name=''):
        typ = agg.type
        try:
            for i in indices:
                typ = typ.elements[i]
        except (AttributeError, IndexError):
            raise TypeError("Can't index at %r in %s" % (list(indices), agg.type))
        super(ExtractValue, self).__init__(parent, typ, 'extractvalue', [agg], name=name)
        self.aggregate = agg
        self.indices = indices

    def descr(self, buf):
        indices = [str(i) for i in self.indices]
        buf.append('extractvalue {0} {1}, {2} {3}\n'.format(self.aggregate.type, self.aggregate.get_reference(), ', '.join(indices), self._stringify_metadata(leading_comma=True)))