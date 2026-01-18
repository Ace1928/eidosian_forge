import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
class TestExternalGreyBoxBlockWithReferences(unittest.TestCase):
    """
    Tests for ExternalGreyBoxBlock with existing variables used
    as inputs and outputs
    """

    def _create_pressure_drop_model(self):
        """
        Create a Pyomo model with pure ExternalGreyBoxModel embedded.
        """
        m = pyo.ConcreteModel()
        m.Pin = pyo.Var()
        m.c = pyo.Var()
        m.F = pyo.Var()
        m.P2 = pyo.Var()
        m.Pout = pyo.Var()
        m.Pin_con = pyo.Constraint(expr=m.Pin == 5.0)
        m.c_con = pyo.Constraint(expr=m.c == 1.0)
        m.F_con = pyo.Constraint(expr=m.F == 10.0)
        m.P2_con = pyo.Constraint(expr=m.P2 <= 5.0)
        m.obj = pyo.Objective(expr=(m.Pout - 3.0) ** 2)
        cons = [m.c_con, m.F_con, m.Pin_con, m.P2_con]
        inputs = [m.Pin, m.c, m.F]
        outputs = [m.P2, m.Pout]
        ex_model = PressureDropTwoOutputsWithHessian()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model, inputs=inputs, outputs=outputs)
        return m

    def test_pressure_drop_model(self):
        m = self._create_pressure_drop_model()
        cons = [m.c_con, m.F_con, m.Pin_con, m.P2_con]
        inputs = [m.Pin, m.c, m.F]
        outputs = [m.P2, m.Pout]
        pyomo_variables = list(m.component_data_objects(pyo.Var))
        pyomo_constraints = list(m.component_data_objects(pyo.Constraint))
        self.assertEqual(len(pyomo_variables), len(inputs) + len(outputs))
        self.assertEqual(len(pyomo_constraints), len(cons))
        self.assertIs(m.egb.inputs.ctype, pyo.Var)
        self.assertIs(m.egb.outputs.ctype, pyo.Var)
        self.assertEqual(len(m.egb.inputs), len(inputs))
        self.assertEqual(len(m.egb.outputs), len(outputs))
        for i in range(len(inputs)):
            self.assertIs(inputs[i], m.egb.inputs[i])
        for i in range(len(outputs)):
            self.assertIs(outputs[i], m.egb.outputs[i])

    def test_pressure_drop_model_nlp(self):
        m = self._create_pressure_drop_model()
        cons = [m.c_con, m.F_con, m.Pin_con, m.P2_con]
        inputs = [m.Pin, m.c, m.F]
        outputs = [m.P2, m.Pout]
        nlp = PyomoNLPWithGreyBoxBlocks(m)
        n_primals = len(inputs) + len(outputs)
        n_eq_con = len(cons) + len(outputs)
        self.assertEqual(nlp.n_primals(), n_primals)
        self.assertEqual(nlp.n_constraints(), n_eq_con)
        constraint_names = ['c_con', 'F_con', 'Pin_con', 'P2_con', 'egb.output_constraints[P2]', 'egb.output_constraints[Pout]']
        primals = inputs + outputs
        nlp_constraints = nlp.constraint_names()
        nlp_vars = nlp.primals_names()
        con_idx_map = {}
        for name in constraint_names:
            con_idx_map[name] = nlp_constraints.index(name)
        var_idx_map = ComponentMap()
        for var in primals:
            name = var.name
            var_idx_map[var] = nlp_vars.index(name)
        incident_vars = {con.name: list(identify_variables(con.expr)) for con in cons}
        incident_vars['egb.output_constraints[P2]'] = inputs + [outputs[0]]
        incident_vars['egb.output_constraints[Pout]'] = inputs + [outputs[1]]
        expected_nonzeros = set()
        for con, varlist in incident_vars.items():
            i = con_idx_map[con]
            for var in varlist:
                j = var_idx_map[var]
                expected_nonzeros.add((i, j))
        self.assertEqual(len(expected_nonzeros), nlp.nnz_jacobian())
        jac = nlp.evaluate_jacobian()
        for i, j in zip(jac.row, jac.col):
            self.assertIn((i, j), expected_nonzeros)