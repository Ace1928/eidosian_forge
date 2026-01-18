import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
@unittest.skipIf(not solver_available, 'The solver is not available.')
class SOSProblem_indexed(object):
    """Test indexed SOS using a single pyomo Var component."""

    def verify(self, model, sos, exp_res, abs_tol, use_rule, case, show_output: bool=False):
        """Make sure the outcome is as expected."""
        opt = pyo.SolverFactory(solver_name)
        problem = model.create_instance()
        opt.solve(problem, tee=show_output)
        assert len(problem.mysos) != 0
        assert math.isclose(pyo.value(problem.OBJ), exp_res, abs_tol=abs_tol)

    def do_it(self, test_number):
        sos, exp_res, abs_tol, use_rule, case = self.test_vectors[test_number]
        model = self.set_problem_up(case=case, n=sos, use_rule=use_rule)
        self.verify(model=model, sos=sos, exp_res=exp_res, abs_tol=abs_tol, use_rule=use_rule, case=case)
    test_vectors = [(1, -0.075, 0.001, True, 0), (1, -0.075, 0.001, False, 0), (2, 1.1, 0.001, True, 0), (2, 1.1, 0.001, False, 0), (1, -0.075, 0.001, True, 1), (1, -0.075, 0.001, False, 1), (2, 1.1, 0.001, True, 1), (2, 1.1, 0.001, False, 1), (1, -0.075, 0.001, True, 2), (1, -0.075, 0.001, False, 2), (2, 1.1, 0.001, True, 2), (2, 1.1, 0.001, False, 2), (1, -0.075, 0.001, True, 3), (1, -0.075, 0.001, False, 3), (2, 1.1, 0.001, True, 3), (2, 1.1, 0.001, False, 3), (1, -0.075, 0.001, True, 4), (1, -0.075, 0.001, False, 4), (2, 1.1, 0.001, True, 4), (2, 1.1, 0.001, False, 4)]

    def set_problem_up(self, case: int == 0, n: int=1, use_rule: bool=False):
        """Create the problem."""
        model = pyo.AbstractModel()
        model.E = pyo.Set(initialize=[1])
        model.A = pyo.Set(initialize=[1, 2, 3, 5, 6])
        model.B = pyo.Set(initialize=[2, 4])
        model.x = pyo.Var(model.E, domain=pyo.NonNegativeReals, bounds=(0, 40))
        model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals)
        model.param_cx = pyo.Param(model.E, initialize={1: 1})
        model.param_cy = pyo.Param(model.A, initialize={1: 2, 2: 3, 3: -0.1, 5: 0.5, 6: 4})

        def obj_f(m):
            return sum((m.param_cx[e] * m.x[e] for e in m.E)) + sum((m.param_cy[a] * m.y[a] for a in m.A))
        model.OBJ = pyo.Objective(rule=obj_f)

        def constr_ya_lb(m, a):
            return m.y[a] <= 2
        model.ConstraintYa_lb = pyo.Constraint(model.A, rule=constr_ya_lb)

        def constr_y_lb(m):
            return m.x[1] + m.y[1] + m.y[2] + m.y[5] + m.y[6] >= 0.25
        model.ConstraintY_lb = pyo.Constraint(rule=constr_y_lb)
        if n == 2:

            def constr_y2_lb(m):
                return m.y[2] + m.y[5] + m.y[6] >= 2.1
            model.ConstraintY2_lb = pyo.Constraint(rule=constr_y2_lb)
        if case == 0:
            index = {2: [1, 3], 4: [2, 5, 6]}
            if use_rule:

                def rule_mysos(m, b):
                    return ([m.y[a] for a in index[b]], [i + 1 for i, _ in enumerate(index[b])])
                model.mysos = pyo.SOSConstraint(model.B, rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(model.B, var=model.y, sos=n, index=index)
        elif case == 1:
            model.mysosindex = pyo.Set(model.B, initialize={2: [1, 3], 4: [2, 5, 6]})
            if use_rule:

                def rule_mysos(m, b):
                    return ([m.y[a] for a in m.mysosindex[b]], [i + 1 for i, _ in enumerate(m.mysosindex[b])])
                model.mysos = pyo.SOSConstraint(model.B, rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(model.B, var=model.y, sos=n, index=model.mysosindex)
        elif case == 2:
            index = {2: [1, 3], 4: [2, 5, 6]}
            weights = {1: 25.0, 3: 18.0, 2: 3, 5: 7, 6: 10}
            if use_rule:

                def rule_mysos(m, b):
                    return ([m.y[a] for a in index[b]], [weights[a] for a in index[b]])
                model.mysos = pyo.SOSConstraint(model.B, rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(model.B, var=model.y, sos=n, index=index, weights=weights)
        elif case == 3:
            model.mysosindex = pyo.Set(model.B, initialize={2: [1, 3], 4: [2, 5, 6]})
            model.mysosweights = pyo.Param(model.A, initialize={1: 25.0, 3: 18.0, 2: 3, 5: 7, 6: 10})
            if use_rule:

                def rule_mysos(m, b):
                    return ([m.y[a] for a in m.mysosindex[b]], [m.mysosweights[a] for a in m.mysosindex[b]])
                model.mysos = pyo.SOSConstraint(model.B, rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(model.B, var=model.y, sos=n, index=model.mysosindex, weights=model.mysosweights)
        else:
            raise NotImplementedError
        return model