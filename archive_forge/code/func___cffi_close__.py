import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def __cffi_close__(self):
    backendlib.close_lib()
    self.__dict__.clear()