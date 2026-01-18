from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class InsertValue(Instruction):

    def __init__(self, parent, agg, elem, indices, name=''):
        typ = agg.type
        try:
            for i in indices:
                typ = typ.elements[i]
        except (AttributeError, IndexError):
            raise TypeError("Can't index at %r in %s" % (list(indices), agg.type))
        if elem.type != typ:
            raise TypeError('Can only insert %s at %r in %s: got %s' % (typ, list(indices), agg.type, elem.type))
        super(InsertValue, self).__init__(parent, agg.type, 'insertvalue', [agg, elem], name=name)
        self.aggregate = agg
        self.value = elem
        self.indices = indices

    def descr(self, buf):
        indices = [str(i) for i in self.indices]
        buf.append('insertvalue {0} {1}, {2} {3}, {4} {5}\n'.format(self.aggregate.type, self.aggregate.get_reference(), self.value.type, self.value.get_reference(), ', '.join(indices), self._stringify_metadata(leading_comma=True)))