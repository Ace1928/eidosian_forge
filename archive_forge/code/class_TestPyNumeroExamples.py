from pyomo.contrib.pynumero.dependencies import numpy_available, scipy_available
import pyomo.common.unittest as unittest
import numpy as np
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.linalg.mumps_interface import mumps_available
from pyomo.contrib.pynumero.linalg.scipy_interface import ScipyLU
import pyomo.environ as pe
from pyomo.contrib.pynumero.examples import (
class TestPyNumeroExamples(unittest.TestCase):

    def test_nlp_interface(self):
        nlp_interface.main()

    def test_nlp_interface_2(self):
        nlp_interface_2.main(show_plot=False)

    @unittest.skipIf(not ipopt_available, 'feasibility example requires ipopt')
    def test_feasibility(self):
        is_feasible = feasibility.main()
        self.assertTrue(is_feasible)

    @unittest.skipIf(not mumps_available, 'mumps example needs pymumps')
    def test_mumps_example(self):
        mumps_example.main()

    @unittest.skipIf(not ipopt_available, 'sensitivity example requires ipopt')
    def test_sensitivity(self):
        x_sens, x_correct = sensitivity.main()
        self.assertTrue(np.allclose(x_sens, x_correct, rtol=0.001, atol=0.0001))

    def test_sqp(self):
        obj = sqp.main(ScipyLU(), 10, 10)
        self.assertAlmostEqual(obj, 0.0498346898889612)