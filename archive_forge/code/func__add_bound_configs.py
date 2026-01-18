import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _add_bound_configs(CONFIG):
    """Adds the bound-related configurations.

    Parameters
    ----------
    CONFIG : ConfigBlock
        The specific configurations for MindtPy.
    """
    CONFIG.declare('obj_bound', ConfigValue(default=1000000000000000.0, domain=PositiveFloat, description='Bound applied to the linearization of the objective function if main MIP is unbounded.'))
    CONFIG.declare('continuous_var_bound', ConfigValue(default=10000000000.0, description='Default bound added to unbounded continuous variables in nonlinear constraint if single tree is activated.', domain=PositiveFloat))
    CONFIG.declare('integer_var_bound', ConfigValue(default=1000000000.0, description='Default bound added to unbounded integral variables in nonlinear constraint if single tree is activated.', domain=PositiveFloat))
    CONFIG.declare('initial_bound_coef', ConfigValue(default=0.1, domain=PositiveFloat, description='The coefficient used to approximate the initial primal/dual bound.'))