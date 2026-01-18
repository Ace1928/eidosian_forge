import struct
from llvmlite.ir._utils import _StrCaching
class LiteralStructType(BaseStructType):
    """
    The type of "literal" structs, i.e. structs with a literally-defined
    type (by contrast with IdentifiedStructType).
    """
    null = 'zeroinitializer'

    def __init__(self, elems, packed=False):
        """
        *elems* is a sequence of types to be used as members.
        *packed* controls the use of packed layout.
        """
        self.elements = tuple(elems)
        self.packed = packed

    def _to_string(self):
        return self.structure_repr()

    def __eq__(self, other):
        if isinstance(other, LiteralStructType):
            return self.elements == other.elements

    def __hash__(self):
        return hash(LiteralStructType)