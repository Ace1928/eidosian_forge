from typing import TYPE_CHECKING
import numpy
from .. import registry
from .numpy_ops import NumpyOps
from .ops import Ops
@registry.ops('MPSOps')
class MPSOps(_Ops):
    """Ops class for Metal Performance shaders."""
    name = 'mps'
    xp = numpy