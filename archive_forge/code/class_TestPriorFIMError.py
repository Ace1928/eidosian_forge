from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.common.unittest as unittest
from pyomo.contrib.doe import (
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
@unittest.skipIf(not numpy_available, 'Numpy is not available')
class TestPriorFIMError(unittest.TestCase):

    def test(self):
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        variable_name = 'C'
        indices = {0: ['CA', 'CB', 'CC'], 1: t_control}
        measurements = MeasurementVariables()
        measurements.add_variables(variable_name, indices=indices, time_index_position=1)
        exp_design = DesignVariables()
        var_C = 'CA0'
        indices_C = {0: [0]}
        exp1_C = [5]
        exp_design.add_variables(var_C, indices=indices_C, time_index_position=0, values=exp1_C, lower_bounds=1, upper_bounds=5)
        var_T = 'T'
        indices_T = {0: t_control}
        exp1_T = [470, 300, 300, 300, 300, 300, 300, 300, 300]
        exp_design.add_variables(var_T, indices=indices_T, time_index_position=0, values=exp1_T, lower_bounds=300, upper_bounds=700)
        parameter_dict = {'A1': 1, 'A2': 1, 'E1': 1}
        prior_right = [[0] * 3 for i in range(3)]
        prior_pass = [[0] * 5 for i in range(10)]
        with self.assertRaises(ValueError):
            doe_object = DesignOfExperiments(parameter_dict, exp_design, measurements, create_model, prior_FIM=prior_pass, discretize_model=disc_for_measure)