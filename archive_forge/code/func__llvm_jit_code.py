import ctypes
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on
def _llvm_jit_code(args, expr, signature, callback_type):
    """Create a native code function from a SymPy expression"""
    if callback_type is None:
        jit = LLVMJitCode(signature)
    else:
        jit = LLVMJitCodeCallback(signature)
    jit._create_args(args)
    jit._create_function_base()
    jit._create_param_dict(args)
    strmod = jit._create_function(expr)
    if False:
        print('LLVM IR')
        print(strmod)
    fptr = jit._compile_function(strmod)
    return fptr