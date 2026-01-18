import logging
from pyomo.common import deprecated
from pyomo.core.base import Transformation, TransformationFactory, Var, Suffix, Reals
@TransformationFactory.register('core.fix_integer_vars', doc='Fix all integer variables to their current values')
class FixIntegerVars(Transformation):

    def __init__(self):
        super(FixIntegerVars, self).__init__()

    def _apply_to(self, model, **kwds):
        options = kwds.pop('options', {})
        if kwds.get('undo', options.get('undo', False)):
            for v in model._fixed_integer_vars[None]:
                v.unfix()
            model.del_component('_fixed_integer_vars')
            return
        fixed_vars = []
        _base_model_vars = model.component_data_objects(Var, active=True, descend_into=True)
        for var in _base_model_vars:
            if var.is_integer() and (not var.is_fixed()):
                fixed_vars.append(var)
                var.fix()
        model._fixed_integer_vars = Suffix(direction=Suffix.LOCAL)
        model._fixed_integer_vars[None] = fixed_vars