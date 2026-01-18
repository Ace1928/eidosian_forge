from contextlib import contextmanager
import logging
from math import fabs
import sys
from pyomo.common import timing
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.core import (
from pyomo.core.expr.numvalue import native_types
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.opt import SolverFactory
def _add_bigm_constraint_to_transformed_model(m, constraint, block):
    """Adds the given constraint to the discrete problem model as if it had
    been on the model originally, before the bigm transformation was called.
    Note this method doesn't actually add the constraint to the model, it just
    takes a constraint that has been added and transforms it.

    Also note that this is not a general method: We know several special
    things in the case of adding OA cuts:
    * No one is going to have a bigm Suffix or arg for this cut--we're
    definitely calculating our own value of M.
    * constraint is for sure a ConstraintData--we don't need to handle anything
    else.
    * We know that we originally called bigm with the default arguments to the
    transformation, so we can assume all of those for this as well. (This is
    part of the reason this *isn't* a general method, what to do about this
    generally is a hard question.)

    Parameters
    ----------
    m: Discrete problem model that has been transformed with bigm.
    constraint: Already-constructed ConstraintData somewhere on m
    block: The block that constraint lives on. This Block may or may not be on
           a Disjunct.
    """
    parent_disjunct = block
    if parent_disjunct.ctype is not Disjunct:
        parent_disjunct = _parent_disjunct(block)
    if parent_disjunct is None:
        return
    bigm = TransformationFactory('gdp.bigm')
    bigm._config = bigm.CONFIG()
    bigm._transform_constraint(Reference(constraint), parent_disjunct, None, [], [])
    del bigm._config