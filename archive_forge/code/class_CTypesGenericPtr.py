import ctypes, ctypes.util, operator, sys
from . import model
class CTypesGenericPtr(CTypesData):
    __slots__ = ['_address', '_as_ctype_ptr']
    _automatic_casts = False
    kind = 'pointer'

    @classmethod
    def _newp(cls, init):
        return cls(init)

    @classmethod
    def _cast_from(cls, source):
        if source is None:
            address = 0
        elif isinstance(source, CTypesData):
            address = source._cast_to_integer()
        elif isinstance(source, (int, long)):
            address = source
        else:
            raise TypeError('bad type for cast to %r: %r' % (cls, type(source).__name__))
        return cls._new_pointer_at(address)

    @classmethod
    def _new_pointer_at(cls, address):
        self = cls.__new__(cls)
        self._address = address
        self._as_ctype_ptr = ctypes.cast(address, cls._ctype)
        return self

    def _get_own_repr(self):
        try:
            return self._addr_repr(self._address)
        except AttributeError:
            return '???'

    def _cast_to_integer(self):
        return self._address

    def __nonzero__(self):
        return bool(self._address)
    __bool__ = __nonzero__

    @classmethod
    def _to_ctypes(cls, value):
        if not isinstance(value, CTypesData):
            raise TypeError('unexpected %s object' % type(value).__name__)
        address = value._convert_to_address(cls)
        return ctypes.cast(address, cls._ctype)

    @classmethod
    def _from_ctypes(cls, ctypes_ptr):
        address = ctypes.cast(ctypes_ptr, ctypes.c_void_p).value or 0
        return cls._new_pointer_at(address)

    @classmethod
    def _initialize(cls, ctypes_ptr, value):
        if value:
            ctypes_ptr.contents = cls._to_ctypes(value).contents

    def _convert_to_address(self, BClass):
        if BClass in (self.__class__, None) or BClass._automatic_casts or self._automatic_casts:
            return self._address
        else:
            return CTypesData._convert_to_address(self, BClass)