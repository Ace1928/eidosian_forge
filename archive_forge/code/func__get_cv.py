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
def _get_cv(kelvin=1, gram=1, mol=1):
    Al = Substance.from_formula('Al', data={'DebyeT': 428 * kelvin, 'mass': 26.9815385})
    Be = Substance.from_formula('Be', data={'DebyeT': 1440 * kelvin, 'mass': 9.0121831})

    def einT(s):
        return 0.806 * s.data['DebyeT']
    return {s.name: EinsteinSolid([einT(s), s.mass * gram / mol]) for s in (Al, Be)}