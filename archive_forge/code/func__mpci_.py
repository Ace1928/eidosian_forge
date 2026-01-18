import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
@property
def _mpci_(self):
    return (self._mpi_, mpi_zero)