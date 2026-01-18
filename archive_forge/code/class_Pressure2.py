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
class Pressure2(Pressure1):

    def args_dimensionality(self):
        return ({'amount': 1},)