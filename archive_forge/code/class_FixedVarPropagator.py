from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.transformation import TransformationFactory
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.config import (
from pyomo.common.errors import InfeasibleConstraintException
@TransformationFactory.register('contrib.propagate_fixed_vars', doc='Propagate variable fixing for equalities of type x = y.')
@document_kwargs_from_configdict('CONFIG')
class FixedVarPropagator(IsomorphicTransformation):
    """Propagate variable fixing for equalities of type :math:`x = y`.

    If :math:`x` is fixed and :math:`y` is not fixed, then this transformation
    will fix :math:`y` to the value of :math:`x`.

    This transformation can also be performed as a temporary transformation,
    whereby the transformed variables are saved and can be later unfixed.

    Keyword arguments below are specified for the ``apply_to`` and
    ``create_using`` functions.

    """
    CONFIG = ConfigBlock()
    CONFIG.declare('tmp', ConfigValue(default=False, domain=bool, description='True to store the set of transformed variables and their old states so that they can be later restored.'))

    def _apply_to(self, instance, **kwds):
        config = self.CONFIG(kwds)
        if config.tmp and (not hasattr(instance, '_tmp_propagate_fixed')):
            instance._tmp_propagate_fixed = ComponentSet()
        eq_var_map, relevant_vars = _build_equality_set(instance)
        fixed_vars = ComponentSet((v for v in relevant_vars if v.fixed))
        newly_fixed = _detect_fixed_variables(instance)
        if config.tmp:
            instance._tmp_propagate_fixed.update(newly_fixed)
        fixed_vars.update(newly_fixed)
        processed = ComponentSet()
        for v1 in fixed_vars:
            if v1 in processed:
                continue
            eq_set = eq_var_map.get(v1, ComponentSet([v1]))
            for v2 in eq_set:
                if v2.fixed and value(v1) != value(v2):
                    raise InfeasibleConstraintException('Variables {} and {} have conflicting fixed values of {} and {}, but are linked by equality constraints.'.format(v1.name, v2.name, value(v1), value(v2)))
                elif not v2.fixed:
                    v2.fix(value(v1))
                    if config.tmp:
                        instance._tmp_propagate_fixed.add(v2)
            processed.update(eq_set)

    def revert(self, instance):
        """Revert variables fixed by the transformation."""
        for var in instance._tmp_propagate_fixed:
            var.unfix()
        del instance._tmp_propagate_fixed