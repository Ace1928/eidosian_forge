import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
def _test_external_greybox_solve(self, ex_model, hessian_support):
    m = pyo.ConcreteModel()
    m.mu = pyo.Var(bounds=(0, None), initialize=1)
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_model)
    m.ccon = pyo.Constraint(expr=m.egb.inputs['c'] == 128 / (3.14 * 0.0001) * m.mu * m.egb.inputs['F'])
    m.pcon = pyo.Constraint(expr=m.egb.inputs['Pin'] - m.egb.outputs['Pout'] <= 72)
    m.pincon = pyo.Constraint(expr=m.egb.inputs['Pin'] == 100.0)
    m.egb.inputs['Pin'].value = 100
    m.egb.inputs['Pin'].setlb(50)
    m.egb.inputs['Pin'].setub(150)
    m.egb.inputs['c'].value = 2
    m.egb.inputs['c'].setlb(1)
    m.egb.inputs['c'].setub(5)
    m.egb.inputs['F'].value = 3
    m.egb.inputs['F'].setlb(1)
    m.egb.inputs['F'].setub(5)
    m.egb.inputs['P1'].value = 80
    m.egb.inputs['P1'].setlb(10)
    m.egb.inputs['P1'].setub(90)
    m.egb.inputs['P3'].value = 70
    m.egb.inputs['P3'].setlb(20)
    m.egb.inputs['P3'].setub(80)
    m.egb.outputs['P2'].value = 75
    m.egb.outputs['P2'].setlb(15)
    m.egb.outputs['P2'].setub(85)
    m.egb.outputs['Pout'].value = 50
    m.egb.outputs['Pout'].setlb(10)
    m.egb.outputs['Pout'].setub(70)
    m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2 + (m.egb.inputs['F'] - 3) ** 2)
    solver = pyo.SolverFactory('cyipopt')
    if not hessian_support:
        solver.config.options = {'hessian_approximation': 'limited-memory'}
    status = solver.solve(m, tee=False)
    self.assertAlmostEqual(pyo.value(m.egb.inputs['F']), 3.0, places=3)
    self.assertAlmostEqual(pyo.value(m.mu), 1.63542e-06, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.outputs['Pout']), 28.0, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.inputs['Pin']), 100.0, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.inputs['c']), 2.0, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.inputs['P1']), 82.0, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.inputs['P3']), 46.0, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.outputs['P2']), 64.0, places=3)