import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
class LowerYield(object):
    """
    Support class for lowering a particular yield point.
    """

    def __init__(self, lower, yield_point, live_vars):
        self.lower = lower
        self.context = lower.context
        self.builder = lower.builder
        self.genlower = lower.genlower
        self.gentype = self.genlower.gentype
        self.gen_state_ptr = self.genlower.gen_state_ptr
        self.resume_index_ptr = self.genlower.resume_index_ptr
        self.yp = yield_point
        self.inst = self.yp.inst
        self.live_vars = live_vars
        self.live_var_indices = [lower.generator_info.state_vars.index(v) for v in live_vars]

    def lower_yield_suspend(self):
        self.lower.debug_print('# generator suspend')
        for state_index, name in zip(self.live_var_indices, self.live_vars):
            state_slot = cgutils.gep_inbounds(self.builder, self.gen_state_ptr, 0, state_index)
            ty = self.gentype.state_types[state_index]
            fetype = self.lower.typeof(name)
            self.lower._alloca_var(name, fetype)
            val = self.lower.loadvar(name)
            if self.context.enable_nrt:
                self.context.nrt.incref(self.builder, ty, val)
            self.context.pack_value(self.builder, ty, val, state_slot)
        indexval = Constant(self.resume_index_ptr.type.pointee, self.inst.index)
        self.builder.store(indexval, self.resume_index_ptr)
        self.lower.debug_print('# generator suspend end')

    def lower_yield_resume(self):
        self.genlower.create_resumption_block(self.lower, self.inst.index)
        self.lower.debug_print('# generator resume')
        for state_index, name in zip(self.live_var_indices, self.live_vars):
            state_slot = cgutils.gep_inbounds(self.builder, self.gen_state_ptr, 0, state_index)
            ty = self.gentype.state_types[state_index]
            val = self.context.unpack_value(self.builder, ty, state_slot)
            self.lower.storevar(val, name)
            if self.context.enable_nrt:
                self.context.nrt.decref(self.builder, ty, val)
        self.lower.debug_print('# generator resume end')