import os
import textwrap
from io import StringIO
from sympy import __version__ as sympy_version
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic
from sympy.printing.c import c_code_printers
from sympy.printing.codeprinter import AssignmentError
from sympy.printing.fortran import FCodePrinter
from sympy.printing.julia import JuliaCodePrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.printing.rust import RustCodePrinter
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
from sympy.utilities.iterables import is_sequence
def _call_printer(self, routine):
    code_lines = []
    declarations = []
    returns = []
    dereference = []
    for arg in routine.arguments:
        if isinstance(arg, ResultBase) and (not arg.dimensions):
            dereference.append(arg.name)
    for i, result in enumerate(routine.results):
        if isinstance(result, Result):
            assign_to = result.result_var
            returns.append(str(result.result_var))
        else:
            raise CodeGenError('unexpected object in Routine results')
        constants, not_supported, rs_expr = self._printer_method_with_settings('doprint', {'human': False}, result.expr, assign_to=assign_to)
        for name, value in sorted(constants, key=str):
            declarations.append('const %s: f64 = %s;\n' % (name, value))
        for obj in sorted(not_supported, key=str):
            if isinstance(obj, Function):
                name = obj.func
            else:
                name = obj
            declarations.append('// unsupported: %s\n' % name)
        code_lines.append('let %s\n' % rs_expr)
    if len(returns) > 1:
        returns = ['(' + ', '.join(returns) + ')']
    returns.append('\n')
    return declarations + code_lines + returns