from ..common.utils import struct_parse
from .decoder import EHABIBytecodeDecoder
from .constants import EHABI_INDEX_ENTRY_SIZE
from .structs import EHABIStructs
class EHABIEntry(object):
    """ Exception handler abi entry.

        Accessible attributes:

            function_offset:
                Integer.
                None if corrupt. (Reference: CorruptEHABIEntry)

            personality:
                Integer.
                None if corrupt or unwindable. (Reference: CorruptEHABIEntry, CannotUnwindEHABIEntry)
                0/1/2 for ARM personality compact format.
                Others for generic personality.

            bytecode_array:
                Integer array.
                None if corrupt or unwindable or generic personality.
                (Reference: CorruptEHABIEntry, CannotUnwindEHABIEntry, GenericEHABIEntry)

            eh_table_offset:
                Integer.
                Only entries who point to .ARM.extab contains this field, otherwise return None.

            unwindable:
                bool. Whether this function is unwindable.

            corrupt:
                bool. Whether this entry is corrupt.

    """

    def __init__(self, function_offset, personality, bytecode_array, eh_table_offset=None, unwindable=True, corrupt=False):
        self.function_offset = function_offset
        self.personality = personality
        self.bytecode_array = bytecode_array
        self.eh_table_offset = eh_table_offset
        self.unwindable = unwindable
        self.corrupt = corrupt

    def mnmemonic_array(self):
        if self.bytecode_array:
            return EHABIBytecodeDecoder(self.bytecode_array).mnemonic_array
        else:
            return None

    def __repr__(self):
        return '<EHABIEntry function_offset=0x%x, personality=%d, %sbytecode=%s>' % (self.function_offset, self.personality, 'eh_table_offset=0x%x, ' % self.eh_table_offset if self.eh_table_offset else '', self.bytecode_array)