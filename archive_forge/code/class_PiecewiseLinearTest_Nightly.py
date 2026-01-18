import os
from os.path import dirname, abspath, join
import pyomo.common.unittest as unittest
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.fileutils import import_file
from pyomo.core.base import Var
from pyomo.core.base.objective import minimize, maximize
from pyomo.core.base.piecewise import Bound, PWRepn
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases
class PiecewiseLinearTest_Nightly(PW_Tests):
    pass