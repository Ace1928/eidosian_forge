import types
import weakref
from copyreg import dispatch_table
def _deepcopy_method(x, memo):
    return type(x)(x.__func__, deepcopy(x.__self__, memo))