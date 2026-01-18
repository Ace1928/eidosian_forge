from __future__ import absolute_import, division, print_function
import ctypes
from itertools import chain
from . import coretypes as ct
def add_default_types(self):
    """
        Adds all the default datashape types to the symbol table.
        """
    self.dtype.update(no_constructor_types)
    self.dtype_constr.update(constructor_types)
    self.dim.update(dim_no_constructor)
    self.dim_constr.update(dim_constructor)