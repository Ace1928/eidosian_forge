import weakref
import numpy
from .dimensionality import Dimensionality
from . import markup
from .quantity import Quantity, get_conversion_factor
from .registry import unit_registry
from .decorators import memoize, with_doc
class UnitConstant(UnitQuantity):
    _primary_order = 0

    def __init__(self, name, definition=None, symbol=None, u_symbol=None, aliases=[], doc=None):
        return