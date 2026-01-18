import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
class ObjCInstance:
    """Python wrapper for an Objective-C instance."""
    pool = 0
    retained = False
    _cached_objects = {}

    def __new__(cls, object_ptr, cache=True):
        """Create a new ObjCInstance or return a previously created one
        for the given object_ptr which should be an Objective-C id."""
        if not isinstance(object_ptr, c_void_p):
            object_ptr = c_void_p(object_ptr)
        if not object_ptr.value:
            return None
        if cache and object_ptr.value in cls._cached_objects:
            return cls._cached_objects[object_ptr.value]
        objc_instance = super(ObjCInstance, cls).__new__(cls)
        objc_instance.ptr = object_ptr
        objc_instance._as_parameter_ = object_ptr
        class_ptr = c_void_p(objc.object_getClass(object_ptr))
        objc_instance.objc_class = ObjCClass(class_ptr)
        if cache:
            cls._cached_objects[object_ptr.value] = objc_instance
            if _arp_manager.current:
                objc_instance.pool = _arp_manager.current
            else:
                _set_dealloc_observer(object_ptr)
        return objc_instance

    def __repr__(self):
        if self.objc_class.name == b'NSCFString':
            from .cocoalibs import cfstring_to_string
            string = cfstring_to_string(self)
            return '<ObjCInstance %#x: %s (%s) at %s>' % (id(self), self.objc_class.name, string, str(self.ptr.value))
        return '<ObjCInstance %#x: %s at %s>' % (id(self), self.objc_class.name, str(self.ptr.value))

    def __getattr__(self, name):
        """Returns a callable method object with the given name."""
        name = ensure_bytes(name)
        method = self.objc_class.get_instance_method(name)
        if method:
            return ObjCBoundMethod(method, self)
        method = self.objc_class.get_class_method(name)
        if method:
            return ObjCBoundMethod(method, self.objc_class.ptr)
        raise AttributeError('ObjCInstance %s has no attribute %s' % (self.objc_class.name, name))