import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def accessor_enum(name, tp=tp, i=i):
    tp.check_not_partial()
    library.__dict__[name] = tp.enumvalues[i]