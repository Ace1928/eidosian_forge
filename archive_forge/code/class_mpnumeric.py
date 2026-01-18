from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
class mpnumeric(object):
    """Base class for mpf and mpc."""
    __slots__ = []

    def __new__(cls, val):
        raise NotImplementedError