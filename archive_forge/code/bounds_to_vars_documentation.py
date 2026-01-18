from math import fabs
import math
from pyomo.core.base.transformation import TransformationFactory
from pyomo.common.config import (
from pyomo.core.base.constraint import Constraint
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
Change constraints to be a bound on the variable.

    Looks for constraints of form: :math:`k*v + c_1 \leq c_2`. Changes
    variable lower bound on :math:`v` to match :math:`(c_2 - c_1)/k` if it
    results in a tighter bound. Also does the same thing for lower bounds.

    Keyword arguments below are specified for the ``apply_to`` and
    ``create_using`` functions.

    