from ..common.utils import struct_parse
from .decoder import EHABIBytecodeDecoder
from .constants import EHABI_INDEX_ENTRY_SIZE
from .structs import EHABIStructs
class GenericEHABIEntry(EHABIEntry):
    """ This entry is generic model rather than ARM compact model.Attribute #bytecode_array will be None.
    """

    def __init__(self, function_offset, personality):
        super(GenericEHABIEntry, self).__init__(function_offset, personality, bytecode_array=None)

    def __repr__(self):
        return '<GenericEHABIEntry function_offset=0x%x, personality=0x%x>' % (self.function_offset, self.personality)