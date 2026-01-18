import struct
from llvmlite.ir._utils import _StrCaching
class PointerType(Type):
    """
    The type of all pointer values.
    """
    is_pointer = True
    null = 'null'

    def __init__(self, pointee, addrspace=0):
        assert not isinstance(pointee, VoidType)
        self.pointee = pointee
        self.addrspace = addrspace

    def _to_string(self):
        if self.addrspace != 0:
            return '{0} addrspace({1})*'.format(self.pointee, self.addrspace)
        else:
            return '{0}*'.format(self.pointee)

    def __eq__(self, other):
        if isinstance(other, PointerType):
            return (self.pointee, self.addrspace) == (other.pointee, other.addrspace)
        else:
            return False

    def __hash__(self):
        return hash(PointerType)

    def gep(self, i):
        """
        Resolve the type of the i-th element (for getelementptr lookups).
        """
        if not isinstance(i.type, IntType):
            raise TypeError(i.type)
        return self.pointee

    @property
    def intrinsic_name(self):
        return 'p%d%s' % (self.addrspace, self.pointee.intrinsic_name)