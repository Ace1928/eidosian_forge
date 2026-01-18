import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class QP_simple(_BaseTestModel):
    """
    A continuous model with a quadratic objective and linear constraints
    """
    description = 'QP_simple'
    level = ('nightly', 'expensive')
    capabilities = set(['linear', 'quadratic_objective'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description + '.json')

    def _generate_model(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description
        model.a = Param(initialize=1.0)
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=NonNegativeReals)
        model.inactive_obj = Objective(expr=model.y)
        model.inactive_obj.deactivate()
        model.obj = Objective(expr=model.x ** 2 + 3.0 * model.inactive_obj ** 2 + 1.0)
        model.c1 = Constraint(expr=model.a <= model.y)
        model.c2 = Constraint(expr=(2.0, model.x / model.a - model.y, 10))

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = 1
        model.y.value = 1

    def post_solve_test_validation(self, tester, results):
        if tester is None:
            assert results['Solver'][0]['termination condition'] in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)
        else:
            tester.assertIn(results['Solver'][0]['termination condition'], (TerminationCondition.optimal, TerminationCondition.locallyOptimal))