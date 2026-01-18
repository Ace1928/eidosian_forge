from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import a_logger, _DoNothing
def _add_nlp_solver_configs(CONFIG, default_solver):
    CONFIG.declare('nlp_solver', ConfigValue(default=default_solver, description='\n            Nonlinear solver to use. Note that no persisent solvers\n            other than the auto-persistent solvers in the APPSI package are\n            supported.'))
    CONFIG.declare('nlp_solver_args', ConfigBlock(description='\n            Keyword arguments to send to the NLP subsolver solve() invocation', implicit=True))
    CONFIG.declare('minlp_solver', ConfigValue(default='baron', description='\n            Mixed-integer nonlinear solver to use. Note that no persisent solvers\n            other than the auto-persistent solvers in the APPSI package are\n            supported.'))
    CONFIG.declare('minlp_solver_args', ConfigBlock(description='\n            Keyword arguments to send to the MINLP subsolver solve() invocation', implicit=True))
    CONFIG.declare('local_minlp_solver', ConfigValue(default='bonmin', description='\n            Mixed-integer nonlinear solver to use. Note that no persisent solvers\n            other than the auto-persistent solvers in the APPSI package are\n            supported.'))
    CONFIG.declare('local_minlp_solver_args', ConfigBlock(description='\n            Keyword arguments to send to the local MINLP subsolver solve()\n            invocation', implicit=True))
    CONFIG.declare('small_dual_tolerance', ConfigValue(default=1e-08, description='\n            When generating cuts, small duals multiplied by expressions can\n            cause problems. Exclude all duals smaller in absolute value than the\n            following.'))