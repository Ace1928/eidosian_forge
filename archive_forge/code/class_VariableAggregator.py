from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.base import Block, Constraint, VarList, Objective, TransformationFactory
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
import logging
@TransformationFactory.register('contrib.aggregate_vars', doc='Aggregate model variables that are linked by equality constraints.')
class VariableAggregator(IsomorphicTransformation):
    """Aggregate model variables that are linked by equality constraints.

    Before:

    .. math::

        x &= y \\\\
        a &= 2x + 6y + 7 \\\\
        b &= 5y + 6 \\\\

    After:

    .. math::

        z &= x = y \\\\
        a &= 8z + 7 \\\\
        b &= 5z + 6

    .. warning:: TODO: unclear what happens to "capital-E" Expressions at this point in time.

    """

    def _apply_to(self, model, detect_fixed_vars=True):
        """Apply the transformation to the given model."""
        eq_var_map = _build_equality_set(model)
        if detect_fixed_vars:
            _fix_equality_fixed_variables(model)
        model._var_aggregator_info = Block(doc='Holds information for the variable aggregation transformation system.')
        z = model._var_aggregator_info.z = VarList(doc='Aggregated variables.')
        z_to_vars = model._var_aggregator_info.z_to_vars = ComponentMap()
        var_to_z = model._var_aggregator_info.var_to_z = ComponentMap()
        processed_vars = ComponentSet()
        for var, eq_set in sorted(eq_var_map.items(), key=lambda tup: tup[0].name):
            if var in processed_vars:
                continue
            assert var_to_z.get(var, None) is None
            z_agg = z.add()
            z_to_vars[z_agg] = eq_set
            var_to_z.update(ComponentMap(((v, z_agg) for v in eq_set)))
            z_agg.setlb(max_if_not_None((v.lb for v in eq_set if v.has_lb())))
            z_agg.setub(min_if_not_None((v.ub for v in eq_set if v.has_ub())))
            fixed_vars = [v for v in eq_set if v.fixed]
            if fixed_vars:
                if any((var.value != fixed_vars[0].value for var in fixed_vars[1:])):
                    raise ValueError('Aggregate variable for equality set is fixed to multiple different values: %s' % (fixed_vars,))
                z_agg.fix(fixed_vars[0].value)
                if z_agg.has_lb() and z_agg.value < value(z_agg.lb):
                    raise ValueError('Aggregate variable for equality set is fixed to a value less than its lower bound: %s < LB %s' % (z_agg.value, value(z_agg.lb)))
                if z_agg.has_ub() and z_agg.value > value(z_agg.ub):
                    raise ValueError('Aggregate variable for equality set is fixed to a value greater than its upper bound: %s > UB %s' % (z_agg.value, value(z_agg.ub)))
            else:
                values_within_bounds = [v.value for v in eq_set if v.value is not None and (not z_agg.has_lb() or v.value >= value(z_agg.lb)) and (not z_agg.has_ub() or v.value <= value(z_agg.ub))]
                if values_within_bounds:
                    z_agg.set_value(sum(values_within_bounds) / len(values_within_bounds), skip_validation=True)
            processed_vars.update(eq_set)
        substitution_map = {id(var): z_var for var, z_var in var_to_z.items()}
        visitor = ExpressionReplacementVisitor(substitute=substitution_map, descend_into_named_expressions=True, remove_named_expressions=False)
        for constr in model.component_data_objects(ctype=Constraint, active=True):
            orig_body = constr.body
            new_body = visitor.walk_expression(constr.body)
            if orig_body is not new_body:
                constr.set_value((constr.lower, new_body, constr.upper))
        for objective in model.component_data_objects(ctype=Objective, active=True):
            orig_expr = objective.expr
            new_expr = visitor.walk_expression(objective.expr)
            if orig_expr is not new_expr:
                objective.set_value(new_expr)

    def update_variables(self, model):
        """Update the values of the variables that were replaced by aggregates.

        TODO: reduced costs

        """
        datablock = model._var_aggregator_info
        for agg_var in datablock.z.itervalues():
            if not agg_var.stale:
                for var in datablock.z_to_vars[agg_var]:
                    var.stale = True
                    var.set_value(agg_var.value, skip_validation=True)