from pyomo.core.base.var import Var
from pyomo.core.base.transformation import TransformationFactory
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
@TransformationFactory.register('contrib.init_vars_midpoint', doc='Initialize non-fixed variables to the midpoint of their bounds.')
class InitMidpoint(IsomorphicTransformation):
    """Initialize non-fixed variables to the midpoint of their bounds.

    - If the variable does not have bounds, set the value to zero.
    - If the variable is missing one bound, set the value to that of the
      existing bound.
    """

    def _apply_to(self, instance, overwrite=False):
        """Apply the transformation.

        Kwargs:
            overwrite: if False, transformation will not overwrite existing
                variable values.
        """
        for var in instance.component_data_objects(ctype=Var, descend_into=True):
            if var.fixed:
                continue
            if var.value is not None and (not overwrite):
                continue
            if var.lb is None and var.ub is None:
                var.set_value(0)
            elif var.lb is None:
                var.set_value(value(var.ub))
            elif var.ub is None:
                var.set_value(value(var.lb))
            else:
                var.set_value((value(var.lb) + value(var.ub)) / 2.0)