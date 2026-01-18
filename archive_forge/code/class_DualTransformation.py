from pyomo.common.deprecation import deprecated
from pyomo.core import (
from pyomo.repn import generate_standard_repn
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.util import partial, process_canonical_repn
@TransformationFactory.register('core.lagrangian_dual', doc='Create the LP dual model.')
class DualTransformation(IsomorphicTransformation):
    """
    Creates a standard form Pyomo model that is equivalent to another model

    Options
        dual_constraint_suffix      Defaults to _constraint
        dual_variable_prefix        Defaults to p_
        slack_names                 Defaults to auxiliary_slack
        excess_names                Defaults to auxiliary_excess
        lb_names                    Defaults to _lower_bound
        ub_names                    Defaults to _upper_bound
        pos_suffix                  Defaults to _plus
        neg_suffix                  Defaults to _minus
    """

    @deprecated('Use of the pyomo.duality package is deprecated. There are known bugs in pyomo.duality, and we do not recommend the use of this code. Development of dualization capabilities has been shifted to the Pyomo Adversarial Optimization (PAO) library. Please contact William Hart for further details (wehart@sandia.gov).', version='5.6.2')
    def __init__(self, **kwds):
        kwds['name'] = 'linear_dual'
        super(DualTransformation, self).__init__(**kwds)

    def _create_using(self, model, **kwds):
        """
        Transform a model to its Lagrangian dual.
        """
        constraint_suffix = kwds.pop('dual_constraint_suffix', '_constraint')
        variable_prefix = kwds.pop('dual_variable_prefix', 'p_')
        sf_kwds = {}
        sf_kwds['slack_names'] = kwds.pop('slack_names', 'auxiliary_slack')
        sf_kwds['excess_names'] = kwds.pop('excess_names', 'auxiliary_excess')
        sf_kwds['lb_names'] = kwds.pop('lb_names', '_lower_bound')
        sf_kwds['ub_names'] = kwds.pop('ub_names', '_upper_bound')
        sf_kwds['pos_suffix'] = kwds.pop('pos_suffix', '_plus')
        sf_kwds['neg_suffix'] = kwds.pop('neg_suffix', '_minus')
        sf_transform = StandardForm()
        sf = sf_transform(model, **sf_kwds)
        A = _sparse(lambda: _sparse(0))
        b = _sparse(0)
        c = _sparse(0)
        for con_name, con_array in sf.component_map(Constraint, active=True).items():
            for con in (con_array[ndx] for ndx in con_array.index_set()):
                cname = '%s%s' % (variable_prefix, con.local_name)
                body_terms = process_canonical_repn(generate_standard_repn(con.body))
                b[cname] -= body_terms.pop(None, 0)
                row = _sparse(0)
                for vname, coef in body_terms.items():
                    row['%s%s' % (vname, constraint_suffix)] += coef
                lower_terms = process_canonical_repn(generate_standard_repn(con.lower))
                b[cname] += lower_terms.pop(None, 0)
                for vname, coef in lower_terms.items():
                    row['%s%s' % (vname, constraint_suffix)] -= coef
                A[cname] = row
        for obj_name, obj_array in sf.component_map(Objective, active=True).items():
            for obj in (obj_array[ndx] for ndx in obj_array.index_set()):
                terms = process_canonical_repn(generate_standard_repn(obj.expr))
                for name, coef in terms.items():
                    c['%s%s' % (name, constraint_suffix)] += coef * obj_array.sense
        dual = AbstractModel()
        constraint_set_init = []
        for var_name, var_array in sf.component_map(Var, active=True).items():
            for var in (var_array[ndx] for ndx in var_array.index_set()):
                constraint_set_init.append('%s%s' % (var.local_name, constraint_suffix))
        variable_set_init = []
        dual_variable_roots = []
        for con_name, con_array in sf.component_map(Constraint, active=True).items():
            for con in (con_array[ndx] for ndx in con_array.index_set()):
                dual_variable_roots.append(con.local_name)
                variable_set_init.append('%s%s' % (variable_prefix, con.local_name))
        dual.var_set = Set(initialize=variable_set_init)
        dual.con_set = Set(initialize=constraint_set_init)
        dual.vars = Var(dual.var_set)

        def constraintRule(A, c, ndx, model):
            return sum((A[v][ndx] * model.vars[v] for v in model.var_set)) <= c[ndx]
        dual.cons = Constraint(dual.con_set, rule=partial(constraintRule, A, c))

        def objectiveRule(b, model):
            return sum((b[v] * model.vars[v] for v in model.var_set))
        dual.obj = Objective(rule=partial(objectiveRule, b), sense=maximize)
        return dual.create()