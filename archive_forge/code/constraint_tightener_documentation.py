import logging
from pyomo.common import deprecated
from pyomo.core import Constraint, value, TransformationFactory
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.standard_repn import generate_standard_repn
Apply the transformation.