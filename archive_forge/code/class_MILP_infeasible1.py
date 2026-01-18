import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, Binary
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class MILP_infeasible1(_BaseTestModel):
    """
    An infeasible mixed-integer linear program
    """
    description = 'MILP_infeasible1'
    capabilities = set(['linear', 'integer'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.solve_should_fail = True
        self.add_results(self.description + '.json')

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description
        model.x = Var(within=Binary)
        model.y = Var(within=Binary)
        model.z = Var(within=Binary)
        model.o = Objective(expr=-model.x - model.y - model.z)
        model.c1 = Constraint(expr=model.x + model.y <= 1)
        model.c2 = Constraint(expr=model.x + model.z <= 1)
        model.c3 = Constraint(expr=model.y + model.z <= 1)
        model.c4 = Constraint(expr=model.x + model.y + model.z >= 1.5)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = None
        model.y.value = None
        model.z.value = None

    def post_solve_test_validation(self, tester, results):
        if tester is None:
            assert results['Solver'][0]['termination condition'] == TerminationCondition.infeasible
        else:
            tester.assertEqual(results['Solver'][0]['termination condition'], TerminationCondition.infeasible)