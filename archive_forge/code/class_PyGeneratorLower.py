import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
class PyGeneratorLower(BaseGeneratorLower):
    """
    Support class for lowering object mode generators.
    """

    def get_generator_type(self):
        """
        Compute the actual generator type (the generator function's return
        type is simply "pyobject").
        """
        return types.Generator(gen_func=self.func_ir.func_id.func, yield_type=types.pyobject, arg_types=(types.pyobject,) * self.func_ir.arg_count, state_types=(types.pyobject,) * len(self.geninfo.state_vars), has_finalizer=True)

    def box_generator_struct(self, lower, gen_struct):
        """
        Box the raw *gen_struct* as a Python object.
        """
        gen_ptr = cgutils.alloca_once_value(lower.builder, gen_struct)
        return lower.pyapi.from_native_generator(gen_ptr, self.gentype, lower.envarg)

    def init_generator_state(self, lower):
        """
        NULL-initialize all generator state variables, to avoid spurious
        decref's on cleanup.
        """
        lower.builder.store(Constant(self.gen_state_ptr.type.pointee, None), self.gen_state_ptr)

    def lower_finalize_func_body(self, builder, genptr):
        """
        Lower the body of the generator's finalizer: decref all live
        state variables.
        """
        pyapi = self.context.get_python_api(builder)
        resume_index_ptr = self.get_resume_index_ptr(builder, genptr)
        resume_index = builder.load(resume_index_ptr)
        need_cleanup = builder.icmp_signed('>', resume_index, Constant(resume_index.type, 0))
        with cgutils.if_unlikely(builder, need_cleanup):
            gen_state_ptr = self.get_state_ptr(builder, genptr)
            for state_index in range(len(self.gentype.state_types)):
                state_slot = cgutils.gep_inbounds(builder, gen_state_ptr, 0, state_index)
                ty = self.gentype.state_types[state_index]
                val = self.context.unpack_value(builder, ty, state_slot)
                pyapi.decref(val)
        builder.ret_void()