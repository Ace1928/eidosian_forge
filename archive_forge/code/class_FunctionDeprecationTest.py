from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Param, Var, Block, Suffix, value
from pyomo.opt import SolverFactory
from pyomo.dae import ContinuousSet
from pyomo.common.dependencies import scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentMap
from pyomo.core.expr import identify_variables
from pyomo.contrib.sensitivity_toolbox.sens import sipopt, kaug, sensitivity_calculation
import pyomo.contrib.sensitivity_toolbox.examples.parameter as param_ex
import pyomo.contrib.sensitivity_toolbox.examples.parameter_kaug as param_kaug_ex
import pyomo.contrib.sensitivity_toolbox.examples.feedbackController as fc
import pyomo.contrib.sensitivity_toolbox.examples.rangeInequality as ri
import pyomo.contrib.sensitivity_toolbox.examples.HIV_Transmission as hiv
class FunctionDeprecationTest(unittest.TestCase):

    @unittest.skipIf(not opt.available(False), 'ipopt_sens is not available')
    def test_sipopt_deprecated(self):
        m = param_ex.create_model()
        m.perturbed_eta1 = Param(initialize=4.0)
        m.perturbed_eta2 = Param(initialize=1.0)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.sensitivity_toolbox', logging.WARNING):
            sipopt(m, [m.eta1, m.eta1], [m.perturbed_eta1, m.perturbed_eta2], cloneModel=False)
        self.assertIn("DEPRECATED: The sipopt function has been deprecated. Use the sensitivity_calculation() function with method='sipopt' to access", output.getvalue().replace('\n', ' '))

    @unittest.skipIf(not opt.available(False), 'ipopt_sens is not available')
    def test_sipopt_equivalent(self):
        m1 = param_ex.create_model()
        m1.perturbed_eta1 = Param(initialize=4.0)
        m1.perturbed_eta2 = Param(initialize=1.0)
        m2 = param_ex.create_model()
        m2.perturbed_eta1 = Param(initialize=4.0)
        m2.perturbed_eta2 = Param(initialize=1.0)
        m11 = sipopt(m1, [m1.eta1, m1.eta2], [m1.perturbed_eta1, m1.perturbed_eta2], cloneModel=True)
        m22 = sensitivity_calculation('sipopt', m2, [m2.eta1, m2.eta2], [m2.perturbed_eta1, m2.perturbed_eta2], cloneModel=True)
        out1 = StringIO()
        out2 = StringIO()
        m11._SENSITIVITY_TOOLBOX_DATA.constList.pprint(ostream=out1)
        m22._SENSITIVITY_TOOLBOX_DATA.constList.pprint(ostream=out2)
        self.assertMultiLineEqual(out1.getvalue(), out2.getvalue())

    @unittest.skipIf(not opt_kaug.available(False), 'k_aug is not available')
    @unittest.skipIf(not opt_dotsens.available(False), 'dot_sens is not available')
    def test_kaug_deprecated(self):
        m = param_ex.create_model()
        m.perturbed_eta1 = Param(initialize=4.0)
        m.perturbed_eta2 = Param(initialize=1.0)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.sensitivity_toolbox', logging.WARNING):
            kaug(m, [m.eta1, m.eta1], [m.perturbed_eta1, m.perturbed_eta2], cloneModel=False)
        self.assertIn("DEPRECATED: The kaug function has been deprecated. Use the sensitivity_calculation() function with method='k_aug'", output.getvalue().replace('\n', ' '))

    @unittest.skipIf(not opt_kaug.available(False), 'k_aug is not available')
    @unittest.skipIf(not opt_dotsens.available(False), 'dot_sens is not available')
    def test_kaug_equivalent(self):
        m1 = param_ex.create_model()
        m1.perturbed_eta1 = Param(initialize=4.0)
        m1.perturbed_eta2 = Param(initialize=1.0)
        m2 = param_ex.create_model()
        m2.perturbed_eta1 = Param(initialize=4.0)
        m2.perturbed_eta2 = Param(initialize=1.0)
        m11 = kaug(m1, [m1.eta1, m1.eta2], [m1.perturbed_eta1, m1.perturbed_eta2], cloneModel=True)
        m22 = sensitivity_calculation('k_aug', m2, [m2.eta1, m2.eta2], [m2.perturbed_eta1, m2.perturbed_eta2], cloneModel=True)
        out1 = StringIO()
        out2 = StringIO()
        m11._SENSITIVITY_TOOLBOX_DATA.constList.pprint(ostream=out1)
        m22._SENSITIVITY_TOOLBOX_DATA.constList.pprint(ostream=out2)
        self.assertMultiLineEqual(out1.getvalue(), out2.getvalue())