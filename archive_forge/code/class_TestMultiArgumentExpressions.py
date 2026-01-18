import pickle
import os
import io
import sys
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.relational_expr import (
class TestMultiArgumentExpressions(unittest.TestCase):

    def test_double_sided_ineq(self):
        m = ConcreteModel()
        m.s = Set(initialize=[1.0, 2.0, 3.0, 4.0, 5.0])
        m.vmin = Param(m.s, initialize=lambda m, i: i)
        m.vmax = Param(m.s, initialize=lambda m, i: i ** 2)
        m.v = Var(m.s)

        def _con(m, i):
            return inequality(m.vmin[i] ** 2, m.v[i], m.vmax[i] ** 2)
        m.con = Constraint(m.s, rule=_con)
        OUT = io.StringIO()
        for i in m.s:
            OUT.write(str(_con(m, i)))
            OUT.write('\n')
        display(m.con, ostream=OUT)
        reference = '1.0  <=  v[1.0]  <=  1.0\n4.0  <=  v[2.0]  <=  16.0\n9.0  <=  v[3.0]  <=  81.0\n16.0  <=  v[4.0]  <=  256.0\n25.0  <=  v[5.0]  <=  625.0\ncon : Size=5\n    Key : Lower : Body : Upper\n    1.0 :   1.0 : None :   1.0\n    2.0 :   4.0 : None :  16.0\n    3.0 :   9.0 : None :  81.0\n    4.0 :  16.0 : None : 256.0\n    5.0 :  25.0 : None : 625.0\n'
        self.assertEqual(OUT.getvalue(), reference)