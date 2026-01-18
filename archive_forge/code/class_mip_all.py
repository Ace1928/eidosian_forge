import json
import os
from os.path import join
from filecmp import cmp
import pyomo.common.unittest as unittest
import pyomo.common
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import ConcreteModel
from pyomo.opt import ResultsFormat, SolverResults, SolverFactory
class mip_all(mock_all):

    def setUp(self):
        self.do_setup(True)