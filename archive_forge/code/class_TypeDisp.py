import base64
import struct
from os_ken.lib import addrconv
class TypeDisp(object):
    _TYPES = {}
    _REV_TYPES = None
    _UNKNOWN_TYPE = None

    @classmethod
    def register_unknown_type(cls):

        def _register_type(subcls):
            cls._UNKNOWN_TYPE = subcls
            return subcls
        return _register_type

    @classmethod
    def register_type(cls, type_):
        cls._TYPES = cls._TYPES.copy()

        def _register_type(subcls):
            cls._TYPES[type_] = subcls
            cls._REV_TYPES = None
            return subcls
        return _register_type

    @classmethod
    def _lookup_type(cls, type_):
        try:
            return cls._TYPES[type_]
        except KeyError:
            return cls._UNKNOWN_TYPE

    @classmethod
    def _rev_lookup_type(cls, targ_cls):
        if cls._REV_TYPES is None:
            rev = dict(((v, k) for k, v in cls._TYPES.items()))
            cls._REV_TYPES = rev
        return cls._REV_TYPES[targ_cls]