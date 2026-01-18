import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.base.matrix_constraint import MatrixConstraint
class TestMatrixConstraint(unittest.TestCase):

    def test_init(self):
        m = pyo.ConcreteModel()
        m.v = _create_variable_list(3, initialize=1.0)
        data, indices, indptr = _get_csr(3, 3, 0.0)
        lb = [None] * 3
        ub = [None] * 3
        m.c = MatrixConstraint(data, indices, indptr, lb, ub, x=list(m.v.values()))
        self.assertEqual(len(m.c), 3)
        for k, c in m.c.items():
            with self.assertRaises(NotImplementedError):
                c.set_value(m.v[k] == 1)
            self.assertEqual(c.index(), k)
            self.assertEqual(c.strict_upper, False)
            self.assertEqual(c.strict_lower, False)
            self.assertEqual(c.lower, None)
            self.assertEqual(c.upper, None)
            self.assertEqual(c.equality, False)
            self.assertEqual(c.body(), 0)
            self.assertEqual(c(), 0)
            self.assertEqual(c.slack(), float('inf'))
            self.assertEqual(c.lslack(), float('inf'))
            self.assertEqual(c.uslack(), float('inf'))
            self.assertEqual(c.has_lb(), False)
            self.assertEqual(c.has_ub(), False)
        m = pyo.ConcreteModel()
        m.v = _create_variable_list(3, initialize=3)
        data, indices, indptr = _get_csr(2, 3, 1.0)
        m.c = MatrixConstraint(data, indices, indptr, lb=[0] * 2, ub=[2] * 2, x=list(m.v.values()))
        self.assertEqual(len(m.c), 2)
        for k, c in m.c.items():
            with self.assertRaises(NotImplementedError):
                c.set_value(m.v[k] == 1)
            self.assertEqual(c.index(), k)
            self.assertEqual(c.strict_upper, False)
            self.assertEqual(c.strict_lower, False)
            self.assertEqual(c.lower, 0)
            self.assertEqual(c.body(), 9)
            self.assertEqual(c(), 9)
            self.assertEqual(c.upper, 2)
            self.assertEqual(c.equality, False)
        m = pyo.ConcreteModel()
        m.v = _create_variable_list(3, initialize=3)
        data, indices, indptr = _get_csr(2, 3, 1.0)
        m.c = MatrixConstraint(data, indices, indptr, lb=[1] * 2, ub=[1] * 2, x=list(m.v.values()))
        self.assertEqual(len(m.c), 2)
        for k, c in m.c.items():
            with self.assertRaises(NotImplementedError):
                c.set_value(m.v[k] == 1)
            self.assertEqual(c.index(), k)
            self.assertEqual(c.strict_upper, False)
            self.assertEqual(c.strict_lower, False)
            self.assertEqual(c.lower, 1)
            self.assertEqual(c.body(), 9)
            self.assertEqual(c(), 9)
            self.assertEqual(c.upper, 1)
            self.assertEqual(c.equality, True)