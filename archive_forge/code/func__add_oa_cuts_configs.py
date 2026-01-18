import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _add_oa_cuts_configs(CONFIG):
    CONFIG.declare('add_slack', ConfigValue(default=False, description='Whether add slack variable here.slack variables here are used to deal with nonconvex MINLP.', domain=bool))
    CONFIG.declare('max_slack', ConfigValue(default=1000.0, domain=PositiveFloat, description='Maximum slack variable', doc='Maximum slack variable value allowed for the Outer Approximation cuts.'))
    CONFIG.declare('OA_penalty_factor', ConfigValue(default=1000.0, domain=PositiveFloat, description='Outer Approximation slack penalty factor', doc='In the objective function of the Outer Approximation method, the slack variables corresponding to all the constraints get multiplied by this number and added to the objective.'))
    CONFIG.declare('equality_relaxation', ConfigValue(default=False, description='Use dual solution from the NLP solver to add OA cuts for equality constraints.', domain=bool))
    CONFIG.declare('linearize_inactive', ConfigValue(default=False, description='Add OA cuts for inactive constraints.', domain=bool))