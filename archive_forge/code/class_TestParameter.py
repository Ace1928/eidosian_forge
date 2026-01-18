from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.common.unittest as unittest
from pyomo.contrib.doe import (
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
class TestParameter(unittest.TestCase):
    """Test the ScenarioGenerator class, generate_scenario function."""

    def test_setup(self):
        param_dict = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}
        scenario_gene = ScenarioGenerator(param_dict, formula='central', step=0.1)
        parameter_set = scenario_gene.ScenarioData
        self.assertAlmostEqual(parameter_set.eps_abs['A1'], 16.9582, places=1)
        self.assertAlmostEqual(parameter_set.eps_abs['E1'], 1.5554, places=1)
        self.assertEqual(parameter_set.scena_num['A2'], [2, 3])
        self.assertEqual(parameter_set.scena_num['E1'], [4, 5])
        self.assertAlmostEqual(parameter_set.scenario[0]['A1'], 93.2699, places=1)
        self.assertAlmostEqual(parameter_set.scenario[2]['A2'], 408.8895, places=1)
        self.assertAlmostEqual(parameter_set.scenario[-1]['E2'], 13.54, places=1)
        self.assertAlmostEqual(parameter_set.scenario[-2]['E2'], 16.55, places=1)