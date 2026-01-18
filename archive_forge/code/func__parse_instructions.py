import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
def _parse_instructions(self, structs, offset, end_offset):
    """ Parse a list of CFI instructions from self.stream, starting with
            the offset and until (not including) end_offset.
            Return a list of CallFrameInstruction objects.
        """
    instructions = []
    while offset < end_offset:
        opcode = struct_parse(structs.Dwarf_uint8(''), self.stream, offset)
        args = []
        primary = opcode & _PRIMARY_MASK
        primary_arg = opcode & _PRIMARY_ARG_MASK
        if primary == DW_CFA_advance_loc:
            args = [primary_arg]
        elif primary == DW_CFA_offset:
            args = [primary_arg, struct_parse(structs.Dwarf_uleb128(''), self.stream)]
        elif primary == DW_CFA_restore:
            args = [primary_arg]
        elif opcode in (DW_CFA_nop, DW_CFA_remember_state, DW_CFA_restore_state, DW_CFA_AARCH64_negate_ra_state):
            args = []
        elif opcode == DW_CFA_set_loc:
            args = [struct_parse(structs.Dwarf_target_addr(''), self.stream)]
        elif opcode == DW_CFA_advance_loc1:
            args = [struct_parse(structs.Dwarf_uint8(''), self.stream)]
        elif opcode == DW_CFA_advance_loc2:
            args = [struct_parse(structs.Dwarf_uint16(''), self.stream)]
        elif opcode == DW_CFA_advance_loc4:
            args = [struct_parse(structs.Dwarf_uint32(''), self.stream)]
        elif opcode in (DW_CFA_offset_extended, DW_CFA_register, DW_CFA_def_cfa, DW_CFA_val_offset):
            args = [struct_parse(structs.Dwarf_uleb128(''), self.stream), struct_parse(structs.Dwarf_uleb128(''), self.stream)]
        elif opcode in (DW_CFA_restore_extended, DW_CFA_undefined, DW_CFA_same_value, DW_CFA_def_cfa_register, DW_CFA_def_cfa_offset):
            args = [struct_parse(structs.Dwarf_uleb128(''), self.stream)]
        elif opcode == DW_CFA_def_cfa_offset_sf:
            args = [struct_parse(structs.Dwarf_sleb128(''), self.stream)]
        elif opcode == DW_CFA_def_cfa_expression:
            args = [struct_parse(structs.Dwarf_dw_form['DW_FORM_block'], self.stream)]
        elif opcode in (DW_CFA_expression, DW_CFA_val_expression):
            args = [struct_parse(structs.Dwarf_uleb128(''), self.stream), struct_parse(structs.Dwarf_dw_form['DW_FORM_block'], self.stream)]
        elif opcode in (DW_CFA_offset_extended_sf, DW_CFA_def_cfa_sf, DW_CFA_val_offset_sf):
            args = [struct_parse(structs.Dwarf_uleb128(''), self.stream), struct_parse(structs.Dwarf_sleb128(''), self.stream)]
        elif opcode == DW_CFA_GNU_args_size:
            args = [struct_parse(structs.Dwarf_uleb128(''), self.stream)]
        else:
            dwarf_assert(False, 'Unknown CFI opcode: 0x%x' % opcode)
        instructions.append(CallFrameInstruction(opcode=opcode, args=args))
        offset = self.stream.tell()
    return instructions