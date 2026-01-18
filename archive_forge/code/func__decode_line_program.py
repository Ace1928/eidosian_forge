import os
import copy
from collections import namedtuple
from ..common.utils import struct_parse, dwarf_assert
from .constants import *
def _decode_line_program(self):
    entries = []
    state = LineState(self.header['default_is_stmt'])

    def add_entry_new_state(cmd, args, is_extended=False):
        entries.append(LineProgramEntry(cmd, is_extended, args, copy.copy(state)))
        state.discriminator = 0
        state.basic_block = False
        state.prologue_end = False
        state.epilogue_begin = False

    def add_entry_old_state(cmd, args, is_extended=False):
        entries.append(LineProgramEntry(cmd, is_extended, args, None))
    offset = self.program_start_offset
    while offset < self.program_end_offset:
        opcode = struct_parse(self.structs.Dwarf_uint8(''), self.stream, offset)
        if opcode >= self.header['opcode_base']:
            maximum_operations_per_instruction = self['maximum_operations_per_instruction']
            adjusted_opcode = opcode - self['opcode_base']
            operation_advance = adjusted_opcode // self['line_range']
            address_addend = self['minimum_instruction_length'] * ((state.op_index + operation_advance) // maximum_operations_per_instruction)
            state.address += address_addend
            state.op_index = (state.op_index + operation_advance) % maximum_operations_per_instruction
            line_addend = self['line_base'] + adjusted_opcode % self['line_range']
            state.line += line_addend
            add_entry_new_state(opcode, [line_addend, address_addend, state.op_index])
        elif opcode == 0:
            inst_len = struct_parse(self.structs.Dwarf_uleb128(''), self.stream)
            ex_opcode = struct_parse(self.structs.Dwarf_uint8(''), self.stream)
            if ex_opcode == DW_LNE_end_sequence:
                state.end_sequence = True
                state.is_stmt = 0
                add_entry_new_state(ex_opcode, [], is_extended=True)
                state = LineState(self.header['default_is_stmt'])
            elif ex_opcode == DW_LNE_set_address:
                operand = struct_parse(self.structs.Dwarf_target_addr(''), self.stream)
                state.address = operand
                add_entry_old_state(ex_opcode, [operand], is_extended=True)
            elif ex_opcode == DW_LNE_define_file:
                operand = struct_parse(self.structs.Dwarf_lineprog_file_entry, self.stream)
                self['file_entry'].append(operand)
                add_entry_old_state(ex_opcode, [operand], is_extended=True)
            elif ex_opcode == DW_LNE_set_discriminator:
                operand = struct_parse(self.structs.Dwarf_uleb128(''), self.stream)
                state.discriminator = operand
            else:
                self.stream.seek(inst_len - 1, os.SEEK_CUR)
        elif opcode == DW_LNS_copy:
            add_entry_new_state(opcode, [])
        elif opcode == DW_LNS_advance_pc:
            operand = struct_parse(self.structs.Dwarf_uleb128(''), self.stream)
            address_addend = operand * self.header['minimum_instruction_length']
            state.address += address_addend
            add_entry_old_state(opcode, [address_addend])
        elif opcode == DW_LNS_advance_line:
            operand = struct_parse(self.structs.Dwarf_sleb128(''), self.stream)
            state.line += operand
        elif opcode == DW_LNS_set_file:
            operand = struct_parse(self.structs.Dwarf_uleb128(''), self.stream)
            state.file = operand
            add_entry_old_state(opcode, [operand])
        elif opcode == DW_LNS_set_column:
            operand = struct_parse(self.structs.Dwarf_uleb128(''), self.stream)
            state.column = operand
            add_entry_old_state(opcode, [operand])
        elif opcode == DW_LNS_negate_stmt:
            state.is_stmt = not state.is_stmt
            add_entry_old_state(opcode, [])
        elif opcode == DW_LNS_set_basic_block:
            state.basic_block = True
            add_entry_old_state(opcode, [])
        elif opcode == DW_LNS_const_add_pc:
            adjusted_opcode = 255 - self['opcode_base']
            address_addend = adjusted_opcode // self['line_range'] * self['minimum_instruction_length']
            state.address += address_addend
            add_entry_old_state(opcode, [address_addend])
        elif opcode == DW_LNS_fixed_advance_pc:
            operand = struct_parse(self.structs.Dwarf_uint16(''), self.stream)
            state.address += operand
            add_entry_old_state(opcode, [operand])
        elif opcode == DW_LNS_set_prologue_end:
            state.prologue_end = True
            add_entry_old_state(opcode, [])
        elif opcode == DW_LNS_set_epilogue_begin:
            state.epilogue_begin = True
            add_entry_old_state(opcode, [])
        elif opcode == DW_LNS_set_isa:
            operand = struct_parse(self.structs.Dwarf_uleb128(''), self.stream)
            state.isa = operand
            add_entry_old_state(opcode, [operand])
        else:
            dwarf_assert(False, 'Invalid standard line program opcode: %s' % (opcode,))
        offset = self.stream.tell()
    return entries