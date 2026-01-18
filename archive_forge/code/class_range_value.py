from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
class range_value(any_value):
    value = Range(-1, 2000000000)