import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def _verification_error(self, msg):
    raise VerificationError(msg)