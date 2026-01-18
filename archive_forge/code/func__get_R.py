from .._util import get_backend
from ..util.regression import least_squares
from ..util.pyutil import defaultnamedtuple
from ..units import default_constants, default_units, format_string, patched_numpy
def _get_R(constants=None, units=None):
    if constants is None:
        R = 8.314472
        if units is not None:
            J = units.Joule
            K = units.Kelvin
            mol = units.mol
            R *= J / mol / K
    else:
        R = constants.molar_gas_constant.simplified
    return R