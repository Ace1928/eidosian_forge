import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
class UnknownFloatType(BasePrimitiveType):
    _attrs_ = ('name',)

    def __init__(self, name):
        self.name = name
        self.c_name_with_marker = name + '&'

    def build_backend_type(self, ffi, finishlist):
        raise NotImplementedError("float type '%s' can only be used after compilation" % self.name)