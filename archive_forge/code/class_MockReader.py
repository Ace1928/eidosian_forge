import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
class MockReader(AbstractResultsReader):

    def __init__(self, name=None):
        AbstractResultsReader.__init__(self, name)