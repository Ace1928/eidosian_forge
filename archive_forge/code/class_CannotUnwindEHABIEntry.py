from ..common.utils import struct_parse
from .decoder import EHABIBytecodeDecoder
from .constants import EHABI_INDEX_ENTRY_SIZE
from .structs import EHABIStructs
class CannotUnwindEHABIEntry(EHABIEntry):
    """ This function cannot be unwind. Attribute #unwindable will be False.
    """

    def __init__(self, function_offset):
        super(CannotUnwindEHABIEntry, self).__init__(function_offset, personality=None, bytecode_array=None, unwindable=False)

    def __repr__(self):
        return '<CannotUnwindEHABIEntry function_offset=0x%x>' % self.function_offset