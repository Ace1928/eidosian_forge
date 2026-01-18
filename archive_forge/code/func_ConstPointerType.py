import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def ConstPointerType(totype):
    return PointerType(totype, Q_CONST)