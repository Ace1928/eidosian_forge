import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
def _get_SpecialFraction_rsys(k, kprime):
    ratex = SpecialFraction([k, kprime], ['k_HBr', 'kprime_HBr'])
    rxn = Reaction({'H2': 1, 'Br2': 1}, {'HBr': 2}, ratex)
    return ReactionSystem([rxn], 'H2 Br2 HBr')