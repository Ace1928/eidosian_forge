from collections import namedtuple
from io import BytesIO
from ..common.utils import struct_parse, bytelist2string, read_blob
from ..common.exceptions import DWARFError
def _init_dispatch_table(structs):
    """Creates a dispatch table for parsing args of an op.

    Returns a dict mapping opcode to a function. The function accepts a stream
    and return a list of parsed arguments for the opcode from the stream;
    the stream is advanced by the function as needed.
    """
    table = {}

    def add(opcode_name, func):
        table[DW_OP_name2opcode[opcode_name]] = func

    def parse_noargs():
        return lambda stream: []

    def parse_op_addr():
        return lambda stream: [struct_parse(structs.Dwarf_target_addr(''), stream)]

    def parse_arg_struct(arg_struct):
        return lambda stream: [struct_parse(arg_struct, stream)]

    def parse_arg_struct2(arg1_struct, arg2_struct):
        return lambda stream: [struct_parse(arg1_struct, stream), struct_parse(arg2_struct, stream)]

    def parse_nestedexpr():

        def parse(stream):
            size = struct_parse(structs.Dwarf_uleb128(''), stream)
            nested_expr_blob = read_blob(stream, size)
            return [DWARFExprParser(structs).parse_expr(nested_expr_blob)]
        return parse

    def parse_blob():
        return lambda stream: [read_blob(stream, struct_parse(structs.Dwarf_uleb128(''), stream))]

    def parse_typedblob():
        return lambda stream: [struct_parse(structs.Dwarf_uleb128(''), stream), read_blob(stream, struct_parse(structs.Dwarf_uint8(''), stream))]

    def parse_wasmloc():

        def parse(stream):
            op = struct_parse(structs.Dwarf_uint8(''), stream)
            if 0 <= op <= 2:
                return [op, struct_parse(structs.Dwarf_uleb128(''), stream)]
            elif op == 3:
                return [op, struct_parse(structs.Dwarf_uint32(''), stream)]
            else:
                raise DWARFError('Unknown operation code in DW_OP_WASM_location: %d' % (op,))
        return parse
    add('DW_OP_addr', parse_op_addr())
    add('DW_OP_addrx', parse_arg_struct(structs.Dwarf_uleb128('')))
    add('DW_OP_const1u', parse_arg_struct(structs.Dwarf_uint8('')))
    add('DW_OP_const1s', parse_arg_struct(structs.Dwarf_int8('')))
    add('DW_OP_const2u', parse_arg_struct(structs.Dwarf_uint16('')))
    add('DW_OP_const2s', parse_arg_struct(structs.Dwarf_int16('')))
    add('DW_OP_const4u', parse_arg_struct(structs.Dwarf_uint32('')))
    add('DW_OP_const4s', parse_arg_struct(structs.Dwarf_int32('')))
    add('DW_OP_const8u', parse_arg_struct(structs.Dwarf_uint64('')))
    add('DW_OP_const8s', parse_arg_struct(structs.Dwarf_int64('')))
    add('DW_OP_constu', parse_arg_struct(structs.Dwarf_uleb128('')))
    add('DW_OP_consts', parse_arg_struct(structs.Dwarf_sleb128('')))
    add('DW_OP_pick', parse_arg_struct(structs.Dwarf_uint8('')))
    add('DW_OP_plus_uconst', parse_arg_struct(structs.Dwarf_uleb128('')))
    add('DW_OP_bra', parse_arg_struct(structs.Dwarf_int16('')))
    add('DW_OP_skip', parse_arg_struct(structs.Dwarf_int16('')))
    for opname in ['DW_OP_deref', 'DW_OP_dup', 'DW_OP_drop', 'DW_OP_over', 'DW_OP_swap', 'DW_OP_swap', 'DW_OP_rot', 'DW_OP_xderef', 'DW_OP_abs', 'DW_OP_and', 'DW_OP_div', 'DW_OP_minus', 'DW_OP_mod', 'DW_OP_mul', 'DW_OP_neg', 'DW_OP_not', 'DW_OP_or', 'DW_OP_plus', 'DW_OP_shl', 'DW_OP_shr', 'DW_OP_shra', 'DW_OP_xor', 'DW_OP_eq', 'DW_OP_ge', 'DW_OP_gt', 'DW_OP_le', 'DW_OP_lt', 'DW_OP_ne', 'DW_OP_nop', 'DW_OP_push_object_address', 'DW_OP_form_tls_address', 'DW_OP_call_frame_cfa', 'DW_OP_stack_value', 'DW_OP_GNU_push_tls_address', 'DW_OP_GNU_uninit']:
        add(opname, parse_noargs())
    for n in range(0, 32):
        add('DW_OP_lit%s' % n, parse_noargs())
        add('DW_OP_reg%s' % n, parse_noargs())
        add('DW_OP_breg%s' % n, parse_arg_struct(structs.Dwarf_sleb128('')))
    add('DW_OP_fbreg', parse_arg_struct(structs.Dwarf_sleb128('')))
    add('DW_OP_regx', parse_arg_struct(structs.Dwarf_uleb128('')))
    add('DW_OP_bregx', parse_arg_struct2(structs.Dwarf_uleb128(''), structs.Dwarf_sleb128('')))
    add('DW_OP_piece', parse_arg_struct(structs.Dwarf_uleb128('')))
    add('DW_OP_bit_piece', parse_arg_struct2(structs.Dwarf_uleb128(''), structs.Dwarf_uleb128('')))
    add('DW_OP_deref_size', parse_arg_struct(structs.Dwarf_int8('')))
    add('DW_OP_xderef_size', parse_arg_struct(structs.Dwarf_int8('')))
    add('DW_OP_call2', parse_arg_struct(structs.Dwarf_uint16('')))
    add('DW_OP_call4', parse_arg_struct(structs.Dwarf_uint32('')))
    add('DW_OP_call_ref', parse_arg_struct(structs.Dwarf_offset('')))
    add('DW_OP_implicit_value', parse_blob())
    add('DW_OP_entry_value', parse_nestedexpr())
    add('DW_OP_const_type', parse_typedblob())
    add('DW_OP_regval_type', parse_arg_struct2(structs.Dwarf_uleb128(''), structs.Dwarf_uleb128('')))
    add('DW_OP_deref_type', parse_arg_struct2(structs.Dwarf_uint8(''), structs.Dwarf_uleb128('')))
    add('DW_OP_implicit_pointer', parse_arg_struct2(structs.Dwarf_offset(''), structs.Dwarf_sleb128('')))
    add('DW_OP_convert', parse_arg_struct(structs.Dwarf_uleb128('')))
    add('DW_OP_GNU_entry_value', parse_nestedexpr())
    add('DW_OP_GNU_const_type', parse_typedblob())
    add('DW_OP_GNU_regval_type', parse_arg_struct2(structs.Dwarf_uleb128(''), structs.Dwarf_uleb128('')))
    add('DW_OP_GNU_deref_type', parse_arg_struct2(structs.Dwarf_uint8(''), structs.Dwarf_uleb128('')))
    add('DW_OP_GNU_implicit_pointer', parse_arg_struct2(structs.Dwarf_offset(''), structs.Dwarf_sleb128('')))
    add('DW_OP_GNU_parameter_ref', parse_arg_struct(structs.Dwarf_offset('')))
    add('DW_OP_GNU_convert', parse_arg_struct(structs.Dwarf_uleb128('')))
    add('DW_OP_WASM_location', parse_wasmloc())
    return table