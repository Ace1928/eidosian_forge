import os
import random
from ..lp_diff import load_and_compare_lp_baseline
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Block, ComponentMap
class Test_CPXLPOrdering_v2(_CPXLPOrdering_Suite, unittest.TestCase):
    _lp_version = 'lp_v2'