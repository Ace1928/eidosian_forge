import sys
import ctypes
from pyglet.util import debug_print
class pInterface(_DummyPointerType, metaclass=_pInterfaceMeta):
    _type_ = Interface

    @classmethod
    def from_param(cls, obj):
        """When dealing with a COMObject, pry a fitting interface out of it"""
        if not isinstance(obj, COMObject):
            return obj
        return obj.as_interface(cls._type_)