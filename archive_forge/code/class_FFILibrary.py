import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
class FFILibrary(object):

    def __getattr__(self, name):
        make_accessor(name)
        return getattr(self, name)

    def __setattr__(self, name, value):
        try:
            property = getattr(self.__class__, name)
        except AttributeError:
            make_accessor(name)
            setattr(self, name, value)
        else:
            property.__set__(self, value)

    def __dir__(self):
        with ffi._lock:
            update_accessors()
            return accessors.keys()

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

    def __cffi_close__(self):
        backendlib.close_lib()
        self.__dict__.clear()