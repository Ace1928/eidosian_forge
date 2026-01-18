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
class OutputArgument(Argument, ResultBase):
    """OutputArgument are always initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol, MatrixSymbol
            The name of this variable.  When used for code generation, this
            might appear, for example, in the prototype of function in the
            argument list.

        result_var : Symbol, Indexed
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".

        expr : object
            The expression that should be output, typically a SymPy
            expression.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimension : sequence containing tupes, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__, self.name, self.result_var, self.expr)
    __repr__ = __str__