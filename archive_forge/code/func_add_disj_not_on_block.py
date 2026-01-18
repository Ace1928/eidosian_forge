from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def add_disj_not_on_block(m):

    def simpdisj_rule(disjunct):
        m = disjunct.model()
        disjunct.c = Constraint(expr=m.a >= 3)
    m.simpledisj = Disjunct(rule=simpdisj_rule)

    def simpledisj2_rule(disjunct):
        m = disjunct.model()
        disjunct.c = Constraint(expr=m.a <= 3.5)
    m.simpledisj2 = Disjunct(rule=simpledisj2_rule)
    m.disjunction2 = Disjunction(expr=[m.simpledisj, m.simpledisj2])
    return m