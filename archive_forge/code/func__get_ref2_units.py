import math
from chempy.chemistry import Reaction
from chempy.util.testing import requires
from chempy.units import units_library, allclose, default_units as u
from ..arrhenius import arrhenius_equation, ArrheniusParam, ArrheniusParamWithUnits
def _get_ref2_units():
    A__s = 10000000000.0
    act_J__mol = 42000.0
    freezing_K = 273.15

    class ValueHolder:
        A = A__s / u.s
        Ea = act_J__mol * u.J / u.mol
        T = freezing_K * u.K
        k = A__s / u.s * math.exp(-act_J__mol / (8.3145 * freezing_K))
    return ValueHolder()