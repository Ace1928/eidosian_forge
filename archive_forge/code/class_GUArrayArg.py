from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
class GUArrayArg(object):

    def __init__(self, context, builder, args, steps, i, step_offset, typ, syms, sym_dim):
        self.context = context
        self.builder = builder
        offset = context.get_constant(types.intp, i)
        data = builder.load(builder.gep(args, [offset], name='data.ptr'), name='data')
        self.data = data
        core_step_ptr = builder.gep(steps, [offset], name='core.step.ptr')
        core_step = builder.load(core_step_ptr)
        if isinstance(typ, types.Array):
            as_scalar = not syms
            if len(syms) != typ.ndim:
                if len(syms) == 0 and typ.ndim == 1:
                    pass
                else:
                    raise TypeError('type and shape signature mismatch for arg #{0}'.format(i + 1))
            ndim = typ.ndim
            shape = [sym_dim[s] for s in syms]
            strides = []
            for j in range(ndim):
                stepptr = builder.gep(steps, [context.get_constant(types.intp, step_offset + j)], name='step.ptr')
                step = builder.load(stepptr)
                strides.append(step)
            ldcls = _ArrayAsScalarArgLoader if as_scalar else _ArrayArgLoader
            self._loader = ldcls(dtype=typ.dtype, ndim=ndim, core_step=core_step, as_scalar=as_scalar, shape=shape, strides=strides)
        else:
            if syms:
                raise TypeError('scalar type {0} given for non scalar argument #{1}'.format(typ, i + 1))
            self._loader = _ScalarArgLoader(dtype=typ, stride=core_step)

    def get_array_at_offset(self, ind):
        return self._loader.load(context=self.context, builder=self.builder, data=self.data, ind=ind)