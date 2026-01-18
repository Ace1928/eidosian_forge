from __future__ import absolute_import, division, print_function
import ctypes
from itertools import chain
from . import coretypes as ct
class TypeSymbolTable:
    """
    This is a class which holds symbols for types and type constructors,
    and is used by the datashape parser to build types during its parsing.
    A TypeSymbolTable sym has four tables, as follows:

    sym.dtype
        Data type symbols with no type constructor.
    sym.dtype_constr
        Data type symbols with a type constructor. This may contain
        symbols also in sym.dtype, e.g. for 'complex' and 'complex[float64]'.
    sym.dim
        Dimension symbols with no type constructor.
    sym.dim_constr
        Dimension symbols with a type constructor.
    """
    __slots__ = ['dtype', 'dtype_constr', 'dim', 'dim_constr']

    def __init__(self, bare=False):
        self.dtype = {}
        self.dtype_constr = {}
        self.dim = {}
        self.dim_constr = {}
        if not bare:
            self.add_default_types()

    def add_default_types(self):
        """
        Adds all the default datashape types to the symbol table.
        """
        self.dtype.update(no_constructor_types)
        self.dtype_constr.update(constructor_types)
        self.dim.update(dim_no_constructor)
        self.dim_constr.update(dim_constructor)