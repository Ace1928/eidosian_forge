import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory
def _qp_model(self):
    m = ConcreteModel(name='test')
    m.x = Var([0, 1, 2])
    m.obj = Objective(expr=m.x[0] + 10 * m.x[1] + 100 * m.x[2] + 1000 * m.x[1] * m.x[2] + 10000 * m.x[0] ** 2 + 10000 * m.x[1] ** 2 + 100000 * m.x[2] ** 2)
    m.c = ConstraintList()
    m.c.add(m.x[0] == 1)
    m.c.add(m.x[1] == 2)
    m.c.add(m.x[2] == 4)
    return m