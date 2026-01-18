from ..common.utils import struct_parse
from .decoder import EHABIBytecodeDecoder
from .constants import EHABI_INDEX_ENTRY_SIZE
from .structs import EHABIStructs
class CorruptEHABIEntry(EHABIEntry):
    """ This entry is corrupt. Attribute #corrupt will be True.
    """

    def __init__(self, reason):
        super(CorruptEHABIEntry, self).__init__(function_offset=None, personality=None, bytecode_array=None, corrupt=True)
        self.reason = reason

    def __repr__(self):
        return '<CorruptEHABIEntry reason=%s>' % self.reason