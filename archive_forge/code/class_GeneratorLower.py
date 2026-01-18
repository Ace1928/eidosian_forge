import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
class GeneratorLower(BaseGeneratorLower):
    """
    Support class for lowering nopython generators.
    """

    def get_generator_type(self):
        return self.fndesc.restype

    def box_generator_struct(self, lower, gen_struct):
        return gen_struct

    def lower_finalize_func_body(self, builder, genptr):
        """
        Lower the body of the generator's finalizer: decref all live
        state variables.
        """
        self.debug_print(builder, '# generator: finalize')
        if self.context.enable_nrt:
            args_ptr = self.get_args_ptr(builder, genptr)
            for ty, val in self.arg_packer.load(builder, args_ptr):
                self.context.nrt.decref(builder, ty, val)
        self.debug_print(builder, '# generator: finalize end')
        builder.ret_void()