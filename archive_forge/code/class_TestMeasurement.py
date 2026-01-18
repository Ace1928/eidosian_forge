from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.common.unittest as unittest
from pyomo.contrib.doe import (
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
class TestMeasurement(unittest.TestCase):
    """Test the MeasurementVariables class, specify, add_element, update_variance, check_subset functions."""

    def test_setup(self):
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        t_control2 = [0.2, 0.4, 0.6, 0.8]
        measurements = MeasurementVariables()
        variable_name = 'C'
        indices = {0: ['CA', 'CB', 'CC'], 1: t_control}
        measurements.add_variables(variable_name, indices=indices, time_index_position=1)
        variable_name2 = 'T'
        indices2 = {0: [1, 3, 5], 1: t_control2}
        measurements.add_variables(variable_name2, indices=indices2, time_index_position=1, variance=10)
        self.assertEqual(measurements.variable_names[0], 'C[CA,0]')
        self.assertEqual(measurements.variable_names[1], 'C[CA,0.125]')
        self.assertEqual(measurements.variable_names[-1], 'T[5,0.8]')
        self.assertEqual(measurements.variable_names[-2], 'T[5,0.6]')
        self.assertEqual(measurements.variance['T[5,0.4]'], 10)
        self.assertEqual(measurements.variance['T[5,0.6]'], 10)
        self.assertEqual(measurements.variance['T[5,0.4]'], 10)
        self.assertEqual(measurements.variance['T[5,0.6]'], 10)
        var_names = ['C[CA,0]', 'C[CA,0.125]', 'C[CA,0.875]', 'C[CA,1]', 'C[CB,0]', 'C[CB,0.125]', 'C[CB,0.25]', 'C[CB,0.375]', 'C[CC,0]', 'C[CC,0.125]', 'C[CC,0.25]', 'C[CC,0.375]']
        measurements2 = MeasurementVariables()
        measurements2.set_variable_name_list(var_names)
        self.assertEqual(measurements2.variable_names[1], 'C[CA,0.125]')
        self.assertEqual(measurements2.variable_names[-1], 'C[CC,0.375]')
        self.assertTrue(measurements.check_subset(measurements2))