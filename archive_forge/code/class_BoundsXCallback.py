from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
class BoundsXCallback(BoundsCallback):
    boundsx = True