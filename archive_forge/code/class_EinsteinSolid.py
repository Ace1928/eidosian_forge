import math
from operator import add
from functools import reduce
import pytest
from chempy import Substance
from chempy.units import (
from ..testing import requires
from ..pyutil import defaultkeydict
from .._expr import (
from ..parsing import parsing_library
class EinsteinSolid(HeatCapacity):
    parameter_keys = HeatCapacity.parameter_keys + ('molar_gas_constant',)
    argument_names = ('einstein_temperature', 'molar_mass')

    def __call__(self, variables, backend=math):
        TE, molar_mass = self.all_args(variables, backend=backend)
        T, R = self.all_params(variables, backend=backend)
        molar_c_v = 3 * R * (TE / (2 * T)) ** 2 * backend.sinh(TE / (2 * T)) ** (-2)
        return molar_c_v / molar_mass