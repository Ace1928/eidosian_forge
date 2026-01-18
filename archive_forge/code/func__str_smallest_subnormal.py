import warnings
from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, NaN
from .umath import log10, exp2, nextafter, isnan
@property
def _str_smallest_subnormal(self):
    """Return the string representation of the smallest subnormal."""
    return self._float_to_str(self.smallest_subnormal)