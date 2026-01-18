from math import fabs
from pyomo.core.base.transformation import TransformationFactory
from pyomo.common.collections import ComponentMap
from pyomo.common.config import (
from pyomo.core.base.var import Var
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
Revert variables fixed by the transformation.