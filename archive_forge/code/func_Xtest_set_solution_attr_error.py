import json
import pickle
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available
def Xtest_set_solution_attr_error(self):
    """Create an error with a solution suffix"""
    try:
        self.soln.variable = True
        self.fail("Expected attribute error failure for 'variable'")
    except AttributeError:
        pass