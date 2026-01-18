import warnings
from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, NaN
from .umath import log10, exp2, nextafter, isnan
def _float_to_float(self, value):
    """Converts float to float.

        Parameters
        ----------
        value : float
            value to be converted.
        """
    return _fr1(self._float_conv(value))