from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
class UArrayArg(object):

    def __init__(self, context, builder, args, steps, i, fe_type):
        self.context = context
        self.builder = builder
        self.fe_type = fe_type
        offset = self.context.get_constant(types.intp, i)
        offseted_args = self.builder.load(builder.gep(args, [offset]))
        data_type = context.get_data_type(fe_type)
        self.dataptr = self.builder.bitcast(offseted_args, data_type.as_pointer())
        sizeof = self.context.get_abi_sizeof(data_type)
        self.abisize = self.context.get_constant(types.intp, sizeof)
        offseted_step = self.builder.gep(steps, [offset])
        self.step = self.builder.load(offseted_step)
        self.is_unit_strided = builder.icmp_unsigned('==', self.abisize, self.step)
        self.builder = builder

    def load_direct(self, byteoffset):
        """
        Generic load from the given *byteoffset*.  load_aligned() is
        preferred if possible.
        """
        ptr = cgutils.pointer_add(self.builder, self.dataptr, byteoffset)
        return self.context.unpack_value(self.builder, self.fe_type, ptr)

    def load_aligned(self, ind):
        ptr = self.builder.gep(self.dataptr, [ind])
        return self.context.unpack_value(self.builder, self.fe_type, ptr)

    def store_direct(self, value, byteoffset):
        ptr = cgutils.pointer_add(self.builder, self.dataptr, byteoffset)
        self.context.pack_value(self.builder, self.fe_type, value, ptr)

    def store_aligned(self, value, ind):
        ptr = self.builder.gep(self.dataptr, [ind])
        self.context.pack_value(self.builder, self.fe_type, value, ptr)