import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def accessor_constant(name):
    raise NotImplementedError("non-integer constant '%s' cannot be accessed from a dlopen() library" % (name,))