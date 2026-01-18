import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def __addressof__(self, name):
    if name in library.__dict__:
        return library.__dict__[name]
    if name in FFILibrary.__dict__:
        return addressof_var(name)
    make_accessor(name)
    if name in library.__dict__:
        return library.__dict__[name]
    if name in FFILibrary.__dict__:
        return addressof_var(name)
    raise AttributeError("cffi library has no function or global variable named '%s'" % (name,))