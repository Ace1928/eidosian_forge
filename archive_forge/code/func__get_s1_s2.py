import pytest
from .._solution import Solution, QuantityDict
from ..util.testing import requires
from ..units import magnitude, units_library, to_unitless, default_units as u
def _get_s1_s2():
    s1 = Solution(0.1 * u.dm3, {'CH3OH': 0.1 * u.molar})
    s2 = Solution(0.3 * u.dm3, {'CH3OH': 0.4 * u.molar, 'Na+': 0.002 * u.molar, 'Cl-': 0.002 * u.molar})
    return (s1, s2)