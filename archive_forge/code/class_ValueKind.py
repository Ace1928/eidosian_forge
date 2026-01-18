from ctypes import (POINTER, byref, cast, c_char_p, c_double, c_int, c_size_t,
import enum
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.typeref import TypeRef
class ValueKind(enum.IntEnum):
    argument = 0
    basic_block = 1
    memory_use = 2
    memory_def = 3
    memory_phi = 4
    function = 5
    global_alias = 6
    global_ifunc = 7
    global_variable = 8
    block_address = 9
    constant_expr = 10
    constant_array = 11
    constant_struct = 12
    constant_vector = 13
    undef_value = 14
    constant_aggregate_zero = 15
    constant_data_array = 16
    constant_data_vector = 17
    constant_int = 18
    constant_fp = 19
    constant_pointer_null = 20
    constant_token_none = 21
    metadata_as_value = 22
    inline_asm = 23
    instruction = 24
    poison_value = 25