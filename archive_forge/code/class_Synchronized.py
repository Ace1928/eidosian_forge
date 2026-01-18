import ctypes
import weakref
from . import heap
from . import get_context
from .context import reduction, assert_spawning
class Synchronized(SynchronizedBase):
    value = make_property('value')