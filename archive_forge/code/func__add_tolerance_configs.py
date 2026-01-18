import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _add_tolerance_configs(CONFIG):
    """Adds the tolerance-related configurations.

    Parameters
    ----------
    CONFIG : ConfigBlock
        The specific configurations for MindtPy.
    """
    CONFIG.declare('absolute_bound_tolerance', ConfigValue(default=0.0001, domain=PositiveFloat, description='Bound tolerance', doc='Absolute tolerance for bound feasibility checks.'))
    CONFIG.declare('relative_bound_tolerance', ConfigValue(default=0.001, domain=PositiveFloat, description='Relative bound tolerance', doc='Relative tolerance for bound feasibility checks. :math:`|Primal Bound - Dual Bound| / (1e-10 + |Primal Bound|) <= relative tolerance`'))
    CONFIG.declare('small_dual_tolerance', ConfigValue(default=1e-08, description='When generating cuts, small duals multiplied by expressions can cause problems. Exclude all duals smaller in absolute value than the following.'))
    CONFIG.declare('integer_tolerance', ConfigValue(default=1e-05, description='Tolerance on integral values.'))
    CONFIG.declare('constraint_tolerance', ConfigValue(default=1e-06, description='Tolerance on constraint satisfaction.'))
    CONFIG.declare('variable_tolerance', ConfigValue(default=1e-08, description='Tolerance on variable bounds.'))
    CONFIG.declare('zero_tolerance', ConfigValue(default=1e-08, description='Tolerance on variable equal to zero.'))