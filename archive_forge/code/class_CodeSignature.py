import ctypes
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on
class CodeSignature:

    def __init__(self, ret_type):
        self.ret_type = ret_type
        self.arg_ctypes = []
        self.input_arg = 0
        self.ret_arg = None