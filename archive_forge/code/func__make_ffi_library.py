import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _make_ffi_library(ffi, libname, flags):
    backend = ffi._backend
    backendlib = _load_backend_lib(backend, libname, flags)

    def accessor_function(name):
        key = 'function ' + name
        tp, _ = ffi._parser._declarations[key]
        BType = ffi._get_cached_btype(tp)
        value = backendlib.load_function(BType, name)
        library.__dict__[name] = value

    def accessor_variable(name):
        key = 'variable ' + name
        tp, _ = ffi._parser._declarations[key]
        BType = ffi._get_cached_btype(tp)
        read_variable = backendlib.read_variable
        write_variable = backendlib.write_variable
        setattr(FFILibrary, name, property(lambda self: read_variable(BType, name), lambda self, value: write_variable(BType, name, value)))

    def addressof_var(name):
        try:
            return addr_variables[name]
        except KeyError:
            with ffi._lock:
                if name not in addr_variables:
                    key = 'variable ' + name
                    tp, _ = ffi._parser._declarations[key]
                    BType = ffi._get_cached_btype(tp)
                    if BType.kind != 'array':
                        BType = model.pointer_cache(ffi, BType)
                    p = backendlib.load_function(BType, name)
                    addr_variables[name] = p
            return addr_variables[name]

    def accessor_constant(name):
        raise NotImplementedError("non-integer constant '%s' cannot be accessed from a dlopen() library" % (name,))

    def accessor_int_constant(name):
        library.__dict__[name] = ffi._parser._int_constants[name]
    accessors = {}
    accessors_version = [False]
    addr_variables = {}

    def update_accessors():
        if accessors_version[0] is ffi._cdef_version:
            return
        for key, (tp, _) in ffi._parser._declarations.items():
            if not isinstance(tp, model.EnumType):
                tag, name = key.split(' ', 1)
                if tag == 'function':
                    accessors[name] = accessor_function
                elif tag == 'variable':
                    accessors[name] = accessor_variable
                elif tag == 'constant':
                    accessors[name] = accessor_constant
            else:
                for i, enumname in enumerate(tp.enumerators):

                    def accessor_enum(name, tp=tp, i=i):
                        tp.check_not_partial()
                        library.__dict__[name] = tp.enumvalues[i]
                    accessors[enumname] = accessor_enum
        for name in ffi._parser._int_constants:
            accessors.setdefault(name, accessor_int_constant)
        accessors_version[0] = ffi._cdef_version

    def make_accessor(name):
        with ffi._lock:
            if name in library.__dict__ or name in FFILibrary.__dict__:
                return
            if name not in accessors:
                update_accessors()
                if name not in accessors:
                    raise AttributeError(name)
            accessors[name](name)

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
    if isinstance(libname, basestring):
        try:
            if not isinstance(libname, str):
                libname = libname.encode('utf-8')
            FFILibrary.__name__ = 'FFILibrary_%s' % libname
        except UnicodeError:
            pass
    library = FFILibrary()
    return (library, library.__dict__)