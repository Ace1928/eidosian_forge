import logging
from pyomo.common import deprecated
from pyomo.core.base import Transformation, TransformationFactory, Var, Suffix, Reals
@TransformationFactory.register('core.relax_integer_vars', doc='Relax integer variables to continuous counterparts')
class RelaxIntegerVars(Transformation):

    def __init__(self):
        super(RelaxIntegerVars, self).__init__()

    def _apply_to(self, model, **kwds):
        options = kwds.pop('options', {})
        if kwds.get('undo', options.get('undo', False)):
            for v, d in model._relaxed_integer_vars[None].values():
                bounds = v.bounds
                v.domain = d
                v.setlb(bounds[0])
                v.setub(bounds[1])
            model.del_component('_relaxed_integer_vars')
            return
        descend = kwds.get('transform_deactivated_blocks', options.get('transform_deactivated_blocks', True))
        active = None if descend else True
        relaxed_vars = {}
        _base_model_vars = model.component_data_objects(Var, active=active, descend_into=True)
        for var in _base_model_vars:
            if not var.is_integer():
                continue
            _c = var.parent_component()
            try:
                lb, ub = var.bounds
                _domain = var.domain
                var.domain = Reals
                var.setlb(lb)
                var.setub(ub)
                relaxed_vars[id(var)] = (var, _domain)
            except:
                if id(_c) in relaxed_vars:
                    continue
                _domain = _c.domain
                lb, ub = _c.bounds
                _c.domain = Reals
                _c.setlb(lb)
                _c.setub(ub)
                relaxed_vars[id(_c)] = (_c, _domain)
        model._relaxed_integer_vars = Suffix(direction=Suffix.LOCAL)
        model._relaxed_integer_vars[None] = relaxed_vars