import ctypes
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on
def _create_function(self, expr):
    """Create function body and return LLVM IR"""
    bb_entry = self.fn.append_basic_block('entry')
    builder = ll.IRBuilder(bb_entry)
    lj = LLVMJitCallbackPrinter(self.module, builder, self.fn, func_arg_map=self.param_dict)
    ret = self._convert_expr(lj, expr)
    if self.signature.ret_arg:
        output_fp_ptr = builder.bitcast(self.fn.args[self.signature.ret_arg], ll.PointerType(self.fp_type))
        for i, val in enumerate(ret):
            index = ll.Constant(ll.IntType(32), i)
            output_array_ptr = builder.gep(output_fp_ptr, [index])
            builder.store(val, output_array_ptr)
        builder.ret(ll.Constant(ll.IntType(32), 0))
    else:
        lj.builder.ret(self._wrap_return(lj, ret))
    strmod = str(self.module)
    return strmod