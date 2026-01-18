from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import a_logger, _DoNothing
def _add_mip_solver_configs(CONFIG):
    CONFIG.declare('mip_solver', ConfigValue(default='gurobi', description='\n            Mixed-integer linear solver to use. Note that no persisent solvers\n            other than the auto-persistent solvers in the APPSI package are\n            supported.'))
    CONFIG.declare('mip_solver_args', ConfigBlock(description='\n            Keyword arguments to send to the MILP subsolver solve() invocation', implicit=True))