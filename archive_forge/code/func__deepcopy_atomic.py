import types
import weakref
from copyreg import dispatch_table
def _deepcopy_atomic(x, memo):
    return x