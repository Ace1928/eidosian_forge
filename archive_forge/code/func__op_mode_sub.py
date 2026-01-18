from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def _op_mode_sub(m, t):
    m.disj1[t].c1 = Constraint(expr=m.x[t] == 2)
    m.disj1[t].sub1 = Disjunct()
    m.disj1[t].sub1.c1 = Constraint(expr=m.y[t] == 100)
    m.disj1[t].sub2 = Disjunct()
    m.disj1[t].sub2.c1 = Constraint(expr=m.y[t] == 1000)
    return [m.disj1[t].sub1, m.disj1[t].sub2]