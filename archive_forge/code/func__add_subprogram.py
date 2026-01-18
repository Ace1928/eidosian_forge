import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
def _add_subprogram(self, name, linkagename, line, function, argmap):
    """Emit subprogram metadata
        """
    subp = self._di_subprogram(name, linkagename, line, function, argmap)
    self.subprograms.append(subp)
    return subp