from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import a_logger, _DoNothing
def _init_strategy_deprecation(strategy):
    deprecation_warning("The argument 'init_strategy' has been deprecated in favor of 'init_algorithm.'", version='6.4.2')
    return In(valid_init_strategies)(strategy)