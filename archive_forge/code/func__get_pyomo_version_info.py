import logging
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.collections import Bunch
from pyomo.core.base.block import Block
from pyomo.core.expr import value
from pyomo.core.base.var import Var
from pyomo.core.base.objective import Objective
from pyomo.contrib.pyros.util import time_code
from pyomo.common.modeling import unique_component_name
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.config import pyros_config
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.solve_data import ROSolveResults
from pyomo.contrib.pyros.pyros_algorithm_methods import ROSolver_iterative_solve
from pyomo.core.base import Constraint
from datetime import datetime
def _get_pyomo_version_info():
    """
    Get Pyomo version information.
    """
    import os
    import subprocess
    from pyomo.version import version
    pyomo_version = version
    commit_hash = 'unknown'
    pyros_dir = os.path.join(*os.path.split(__file__)[:-1])
    commit_hash_command_args = ['git', '-C', f'{pyros_dir}', 'rev-parse', '--short', 'HEAD']
    try:
        commit_hash = subprocess.check_output(commit_hash_command_args).decode('ascii').strip()
    except subprocess.CalledProcessError:
        commit_hash = 'unknown'
    return {'Pyomo version': pyomo_version, 'Commit hash': commit_hash}