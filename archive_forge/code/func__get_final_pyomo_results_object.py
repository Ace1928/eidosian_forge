from io import StringIO
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigBlock
from pyomo.common.errors import DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.config_options import _add_common_configs
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt import __version__
from pyomo.contrib.gdpopt.util import (
from pyomo.core.base import Objective, value, minimize, maximize
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt import SolverResults
from pyomo.opt import TerminationCondition as tc
from pyomo.util.model_size import build_model_size_report
def _get_final_pyomo_results_object(self):
    """
        Fill in the results.solver information onto the results object
        """
    results = self.pyomo_results
    results.problem.lower_bound = self.LB
    results.problem.upper_bound = self.UB
    results.solver.iterations = self.iteration
    results.solver.timing = self.timing
    results.solver.user_time = self.timing.total
    results.solver.wallclock_time = self.timing.total
    return results