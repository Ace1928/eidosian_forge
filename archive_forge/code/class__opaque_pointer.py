import ctypes
class _opaque_pointer(ctypes.POINTER(_Opaque)):
    _type_ = _Opaque

    @classmethod
    def from_param(cls, value):
        return ctypes.cast(value, cls)

    @property
    def address(self):
        return ctypes.addressof(self.contents)

    @property
    def as_voidp(self):
        return ctypes.c_voidp(self.address)

    def __hash__(self):
        """Allow these pointers to be used as keys in dictionaries"""
        return self.address