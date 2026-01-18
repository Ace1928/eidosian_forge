import struct
from llvmlite.ir._utils import _StrCaching
class BaseStructType(Aggregate):
    """
    The base type for heterogenous struct types.
    """
    _packed = False

    @property
    def packed(self):
        """
        A boolean attribute that indicates whether the structure uses
        packed layout.
        """
        return self._packed

    @packed.setter
    def packed(self, val):
        self._packed = bool(val)

    def __len__(self):
        assert self.elements is not None
        return len(self.elements)

    def __iter__(self):
        assert self.elements is not None
        return iter(self.elements)

    @property
    def is_opaque(self):
        return self.elements is None

    def structure_repr(self):
        """
        Return the LLVM IR for the structure representation
        """
        ret = '{%s}' % ', '.join([str(x) for x in self.elements])
        return self._wrap_packed(ret)

    def format_constant(self, value):
        itemstring = ', '.join(['{0} {1}'.format(x.type, x.get_reference()) for x in value])
        ret = '{{{0}}}'.format(itemstring)
        return self._wrap_packed(ret)

    def gep(self, i):
        """
        Resolve the type of the i-th element (for getelementptr lookups).

        *i* needs to be a LLVM constant, so that the type can be determined
        at compile-time.
        """
        if not isinstance(i.type, IntType):
            raise TypeError(i.type)
        return self.elements[i.constant]

    def _wrap_packed(self, textrepr):
        """
        Internal helper to wrap textual repr of struct type into packed struct
        """
        if self.packed:
            return '<{}>'.format(textrepr)
        else:
            return textrepr