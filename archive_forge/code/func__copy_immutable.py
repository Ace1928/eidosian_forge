import types
import weakref
from copyreg import dispatch_table
def _copy_immutable(x):
    return x