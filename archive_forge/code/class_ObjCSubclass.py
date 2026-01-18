import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
class ObjCSubclass:
    """Use this to create a subclass of an existing Objective-C class.
    It consists primarily of function decorators which you use to add methods
    to the subclass."""

    def __init__(self, superclass, name, register=True):
        self._imp_table = {}
        self.name = name
        self.objc_cls = create_subclass(superclass, name)
        self._as_parameter_ = self.objc_cls
        if register:
            self.register()

    def register(self):
        """Register the new class with the Objective-C runtime."""
        objc.objc_registerClassPair(self.objc_cls)
        self.objc_metaclass = get_metaclass(self.name)

    def add_ivar(self, varname, vartype):
        """Add instance variable named varname to the subclass.
        varname should be a string.
        vartype is a ctypes type.
        The class must be registered AFTER adding instance variables."""
        return add_ivar(self.objc_cls, varname, vartype)

    def add_method(self, method, name, encoding):
        imp = add_method(self.objc_cls, name, method, encoding)
        self._imp_table[name] = imp

    def add_class_method(self, method, name, encoding):
        imp = add_method(self.objc_metaclass, name, method, encoding)
        self._imp_table[name] = imp

    def rawmethod(self, encoding):
        """Decorator for instance methods without any fancy shenanigans.
        The function must have the signature f(self, cmd, *args)
        where both self and cmd are just pointers to objc objects."""
        encoding = ensure_bytes(encoding)
        typecodes = parse_type_encoding(encoding)
        typecodes.insert(1, b'@:')
        encoding = b''.join(typecodes)

        def decorator(f):
            name = f.__name__.replace('_', ':')
            self.add_method(f, name, encoding)
            return f
        return decorator

    def method(self, encoding):
        """Function decorator for instance methods."""
        encoding = ensure_bytes(encoding)
        typecodes = parse_type_encoding(encoding)
        typecodes.insert(1, b'@:')
        encoding = b''.join(typecodes)

        def decorator(f):

            def objc_method(objc_self, objc_cmd, *args):
                py_self = ObjCInstance(objc_self, True)
                py_self.objc_cmd = objc_cmd
                py_self.retained = True
                args = convert_method_arguments(encoding, args)
                result = f(py_self, *args)
                if isinstance(result, ObjCClass):
                    result = result.ptr.value
                elif isinstance(result, ObjCInstance):
                    result = result.ptr.value
                return result
            name = f.__name__.replace('_', ':')
            self.add_method(objc_method, name, encoding)
            return objc_method
        return decorator

    def classmethod(self, encoding):
        """Function decorator for class methods."""
        encoding = ensure_bytes(encoding)
        typecodes = parse_type_encoding(encoding)
        typecodes.insert(1, b'@:')
        encoding = b''.join(typecodes)

        def decorator(f):

            def objc_class_method(objc_cls, objc_cmd, *args):
                py_cls = ObjCClass(objc_cls)
                py_cls.objc_cmd = objc_cmd
                args = convert_method_arguments(encoding, args)
                result = f(py_cls, *args)
                if isinstance(result, ObjCClass):
                    result = result.ptr.value
                elif isinstance(result, ObjCInstance):
                    result = result.ptr.value
                return result
            name = f.__name__.replace('_', ':')
            self.add_class_method(objc_class_method, name, encoding)
            return objc_class_method
        return decorator