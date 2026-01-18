from functools import reduce
import operator
import math
from llvmlite import ir
import llvmlite.binding as ll
from numba.core.imputils import Registry, lower_cast
from numba.core.typing.npydecl import parse_dtype
from numba.core.datamodel import models
from numba.core import types, cgutils
from numba.np import ufunc_db
from numba.np.npyimpl import register_ufuncs
from .cudadrv import nvvm
from numba import cuda
from numba.cuda import nvvmutils, stubs, errors
from numba.cuda.types import dim3, CUDADispatcher
def _generic_array(context, builder, shape, dtype, symbol_name, addrspace, can_dynsized=False):
    elemcount = reduce(operator.mul, shape, 1)
    dynamic_smem = elemcount <= 0 and can_dynsized and (len(shape) == 1)
    if elemcount <= 0 and (not dynamic_smem):
        raise ValueError('array length <= 0')
    data_model = context.data_model_manager[dtype]
    other_supported_type = isinstance(dtype, (types.Record, types.Boolean)) or isinstance(data_model, models.StructModel) or dtype == types.float16
    if dtype not in types.number_domain and (not other_supported_type):
        raise TypeError('unsupported type: %s' % dtype)
    lldtype = context.get_data_type(dtype)
    laryty = ir.ArrayType(lldtype, elemcount)
    if addrspace == nvvm.ADDRSPACE_LOCAL:
        dataptr = cgutils.alloca_once(builder, laryty, name=symbol_name)
    else:
        lmod = builder.module
        gvmem = cgutils.add_global_variable(lmod, laryty, symbol_name, addrspace)
        align = context.get_abi_sizeof(lldtype)
        gvmem.align = 1 << (align - 1).bit_length()
        if dynamic_smem:
            gvmem.linkage = 'external'
        else:
            gvmem.initializer = ir.Constant(laryty, ir.Undefined)
        dataptr = builder.addrspacecast(gvmem, ir.PointerType(ir.IntType(8)), 'generic')
    targetdata = ll.create_target_data(nvvm.NVVM().data_layout)
    lldtype = context.get_data_type(dtype)
    itemsize = lldtype.get_abi_size(targetdata)
    laststride = itemsize
    rstrides = []
    for i, lastsize in enumerate(reversed(shape)):
        rstrides.append(laststride)
        laststride *= lastsize
    strides = [s for s in reversed(rstrides)]
    kstrides = [context.get_constant(types.intp, s) for s in strides]
    if dynamic_smem:
        get_dynshared_size = ir.InlineAsm(ir.FunctionType(ir.IntType(32), []), 'mov.u32 $0, %dynamic_smem_size;', '=r', side_effect=True)
        dynsmem_size = builder.zext(builder.call(get_dynshared_size, []), ir.IntType(64))
        kitemsize = context.get_constant(types.intp, itemsize)
        kshape = [builder.udiv(dynsmem_size, kitemsize)]
    else:
        kshape = [context.get_constant(types.intp, s) for s in shape]
    ndim = len(shape)
    aryty = types.Array(dtype=dtype, ndim=ndim, layout='C')
    ary = context.make_array(aryty)(context, builder)
    context.populate_array(ary, data=builder.bitcast(dataptr, ary.data.type), shape=kshape, strides=kstrides, itemsize=context.get_constant(types.intp, itemsize), meminfo=None)
    return ary._getvalue()