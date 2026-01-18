from collections import defaultdict, OrderedDict
from itertools import permutations
import math
import pytest
from chempy import Equilibrium, Reaction, ReactionSystem, Substance
from chempy.thermodynamics.expressions import MassActionEq
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.testing import requires
from .test_rates import _get_SpecialFraction_rsys
from ..arrhenius import ArrheniusParam
from ..rates import Arrhenius, MassAction, Radiolytic, RampedTemp
from .._rates import ShiftedTPoly
from ..ode import (
from ..integrated import dimerization_irrev, binary_rev
class RampedRate(Expr):
    argument_names = ('rate_constant', 'ramping_rate')

    def __call__(self, variables, reaction, backend=math):
        rate_constant, ramping_rate = self.all_args(variables, backend=backend)
        return rate_constant * ramping_rate * variables['time']