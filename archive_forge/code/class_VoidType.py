import struct
from llvmlite.ir._utils import _StrCaching
class VoidType(Type):
    """
    The type for empty values (e.g. a function returning no value).
    """

    def _to_string(self):
        return 'void'

    def __eq__(self, other):
        return isinstance(other, VoidType)

    def __hash__(self):
        return hash(VoidType)