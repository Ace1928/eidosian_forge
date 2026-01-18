from itertools import zip_longest
import re
import sys
import os
from os.path import join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import ProblemFormat, ConverterError, convert_problem
from pyomo.common import Executable
class MockArg3(MockArg):

    def valid_problem_types(self):
        return [ProblemFormat.mod]

    def write(self, filename='', format=None, solver_capability=None, io_options={}):
        return (filename, None)