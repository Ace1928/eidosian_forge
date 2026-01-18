import json
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.fileutils import import_file
import os
from os.path import abspath, dirname, normpath, join
@unittest.skipIf(not casadi_available, 'Casadi is not available')
class TestCasadiSimulation(unittest.TestCase, TestSimulationInterface):
    sim_mod = 'casadi'

    def test_ode_example(self):
        tname = 'simulator_ode_example'
        self._test(tname)

    def test_ode_example2(self):
        tname = 'simulator_ode_example'
        self._test_disc_first(tname)

    def test_ode_multindex_example(self):
        tname = 'simulator_ode_multindex_example'
        self._test(tname)

    def test_ode_multindex_example2(self):
        tname = 'simulator_ode_multindex_example'
        self._test_disc_first(tname)

    def test_dae_example(self):
        tname = 'simulator_dae_example'
        self._test(tname)

    def test_dae_example2(self):
        tname = 'simulator_dae_example'
        self._test_disc_first(tname)

    def test_dae_multindex_example(self):
        tname = 'simulator_dae_multindex_example'
        self._test(tname)

    def test_dae_multindex_example2(self):
        tname = 'simulator_dae_multindex_example'
        self._test_disc_first(tname)