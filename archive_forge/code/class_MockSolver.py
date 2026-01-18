import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
class MockSolver(OptSolver):

    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        kwds['doc'] = 'MockSolver Documentation'
        OptSolver.__init__(self, **kwds)