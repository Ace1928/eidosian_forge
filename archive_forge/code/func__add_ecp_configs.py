import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _add_ecp_configs(CONFIG):
    CONFIG.declare('ecp_tolerance', ConfigValue(default=None, domain=PositiveFloat, description='ECP tolerance', doc='Feasibility tolerance used to determine the stopping criterion inthe ECP method. As long as nonlinear constraint are violated for more than this tolerance, the method will keep iterating.'))
    CONFIG.declare('init_strategy', ConfigValue(default='max_binary', domain=In(['rNLP', 'max_binary', 'FP']), description='Initialization strategy', doc='Initialization strategy used by any method. Currently the continuous relaxation of the MINLP (rNLP), solve a maximal covering problem (max_binary), and fix the initial value for the integer variables (initial_binary).'))