from pyomo.contrib.mindtpy import __version__
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_config
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.contrib.mindtpy.config_options import _supported_algorithms
Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        