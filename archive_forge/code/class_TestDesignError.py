from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.common.unittest as unittest
from pyomo.contrib.doe import (
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
class TestDesignError(unittest.TestCase):

    def test(self):
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        exp_design = DesignVariables()
        var_T = 'T'
        indices_T = {0: t_control}
        exp1_T = [470, 300, 300, 300, 300, 300, 300, 300, 300]
        upper_bound = [700, 700, 700, 700, 700, 700, 700, 700, 700, 800]
        lower_bound = [300, 300, 300, 300, 300, 300, 300, 300, 300]
        with self.assertRaises(ValueError):
            exp_design.add_variables(var_T, indices=indices_T, time_index_position=0, values=exp1_T, lower_bounds=lower_bound, upper_bounds=upper_bound)