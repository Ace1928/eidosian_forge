import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _add_goa_configs(CONFIG):
    CONFIG.declare('init_strategy', ConfigValue(default='rNLP', domain=In(['rNLP', 'initial_binary', 'max_binary']), description='Initialization strategy', doc='Initialization strategy used by any method. Currently the continuous relaxation of the MINLP (rNLP), solve a maximal covering problem (max_binary), and fix the initial value for the integer variables (initial_binary).'))