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
class DataType:
    """Holds strings for a certain datatype in different languages."""

    def __init__(self, cname, fname, pyname, jlname, octname, rsname):
        self.cname = cname
        self.fname = fname
        self.pyname = pyname
        self.jlname = jlname
        self.octname = octname
        self.rsname = rsname