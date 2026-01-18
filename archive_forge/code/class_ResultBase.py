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
class ResultBase:
    """Base class for all "outgoing" information from a routine.

    Objects of this class stores a SymPy expression, and a SymPy object
    representing a result variable that will be used in the generated code
    only if necessary.

    """

    def __init__(self, expr, result_var):
        self.expr = expr
        self.result_var = result_var

    def __str__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.expr, self.result_var)
    __repr__ = __str__