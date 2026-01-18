from pyomo.common.dependencies import pandas as pd, pandas_available
import pyomo.common.unittest as unittest
import os
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.scenariocreator as sc
import pyomo.environ as pyo
from pyomo.environ import SolverFactory
@unittest.skipIf(not parmest.parmest_available, 'Cannot test parmest: required dependencies are missing')
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestScenarioSemibatch(unittest.TestCase):

    def setUp(self):
        import pyomo.contrib.parmest.examples.semibatch.semibatch as sb
        import json
        theta_names = ['k1', 'k2', 'E1', 'E2']
        self.fbase = os.path.join(testdir, '..', 'examples', 'semibatch')
        data = []
        for exp_num in range(10):
            fname = 'exp' + str(exp_num + 1) + '.out'
            fullname = os.path.join(self.fbase, fname)
            with open(fullname, 'r') as infile:
                d = json.load(infile)
                data.append(d)
        self.pest = parmest.Estimator(sb.generate_model, data, theta_names)

    def test_semibatch_bootstrap(self):
        scenmaker = sc.ScenarioCreator(self.pest, 'ipopt')
        bootscens = sc.ScenarioSet('Bootstrap')
        numtomake = 2
        scenmaker.ScenariosFromBootstrap(bootscens, numtomake, seed=1134)
        tval = bootscens.ScenarioNumber(0).ThetaVals['k1']
        self.assertAlmostEqual(tval, 20.64, places=1)