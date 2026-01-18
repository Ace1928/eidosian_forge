import ctypes
class VARIANT_BOOL(ctypes._SimpleCData):
    _type_ = 'v'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.value)