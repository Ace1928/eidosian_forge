from .._util import get_backend
from ..util.regression import least_squares
from ..util.pyutil import defaultnamedtuple
from ..units import default_constants, default_units, format_string, patched_numpy
def Ea_over_R(self, constants, units, backend=None):
    return self.Ea / _get_R(constants, units)