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
def _get_symbol(self, s):
    """Returns the symbol as fcode prints it."""
    if self.printer._settings['human']:
        expr_str = self.printer.doprint(s)
    else:
        constants, not_supported, expr_str = self.printer.doprint(s)
        if constants or not_supported:
            raise ValueError('Failed to print %s' % str(s))
    return expr_str.strip()