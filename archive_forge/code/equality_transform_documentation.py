from pyomo.core import TransformationFactory, Var, NonNegativeReals
from pyomo.core.base.misc import create_name
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.util import collectAbstractComponents

        Eliminate inequality constraints.

        Required arguments:

          model The model to transform.

        Optional keyword arguments:

          slack_root  The root name of auxiliary slack variables.
                      Default is 'auxiliary_slack'.
          excess_root The root name of auxiliary slack variables.
                      Default is 'auxiliary_excess'.
          lb_suffix   The suffix applied to converted upper bound constraints
                      Default is '_lower_bound'.
          ub_suffix   The suffix applied to converted lower bound constraints
                      Default is '_upper_bound'.
        