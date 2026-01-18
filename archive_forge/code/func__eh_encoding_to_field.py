import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
@staticmethod
def _eh_encoding_to_field(entry_structs):
    """
        Return a mapping from basic encodings (DW_EH_encoding_flags) the
        corresponding field constructors (for instance
        entry_structs.Dwarf_uint32).
        """
    return {DW_EH_encoding_flags['DW_EH_PE_absptr']: entry_structs.Dwarf_target_addr, DW_EH_encoding_flags['DW_EH_PE_uleb128']: entry_structs.Dwarf_uleb128, DW_EH_encoding_flags['DW_EH_PE_udata2']: entry_structs.Dwarf_uint16, DW_EH_encoding_flags['DW_EH_PE_udata4']: entry_structs.Dwarf_uint32, DW_EH_encoding_flags['DW_EH_PE_udata8']: entry_structs.Dwarf_uint64, DW_EH_encoding_flags['DW_EH_PE_sleb128']: entry_structs.Dwarf_sleb128, DW_EH_encoding_flags['DW_EH_PE_sdata2']: entry_structs.Dwarf_int16, DW_EH_encoding_flags['DW_EH_PE_sdata4']: entry_structs.Dwarf_int32, DW_EH_encoding_flags['DW_EH_PE_sdata8']: entry_structs.Dwarf_int64}