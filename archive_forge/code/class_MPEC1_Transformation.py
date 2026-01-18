import logging
from pyomo.core.base import (
from pyomo.mpec.complementarity import Complementarity
from pyomo.gdp import Disjunct
@TransformationFactory.register('mpec.simple_nonlinear', doc='Nonlinear transformations of complementarity conditions when all variables are non-negative')
class MPEC1_Transformation(Transformation):

    def __init__(self):
        super(MPEC1_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        options = kwds.pop('options', {})
        bound = kwds.pop('mpec_bound', 0.0)
        bound = options.get('mpec_bound', bound)
        instance.mpec_bound = Param(mutable=True, initialize=bound)
        tdata = instance._transformation_data['mpec.simple_nonlinear']
        tdata.compl_cuids = []
        for complementarity in instance.component_objects(Complementarity, active=True, descend_into=(Block, Disjunct), sort=SortComponents.deterministic):
            block = complementarity.parent_block()
            for index in sorted(complementarity.keys()):
                _data = complementarity[index]
                if not _data.active:
                    continue
                _data.to_standard_form()
                _type = getattr(_data.c, '_complementarity_type', 0)
                if _type == 1:
                    _data.ccon = Constraint(expr=(_data.c.body - _data.c.lower) * _data.v <= instance.mpec_bound)
                    del _data.c._complementarity_type
                elif _type == 3:
                    _data.ccon_l = Constraint(expr=(_data.v - _data.v.bounds[0]) * _data.c.body <= instance.mpec_bound)
                    _data.ccon_u = Constraint(expr=(_data.v - _data.v.bounds[1]) * _data.c.body <= instance.mpec_bound)
                    del _data.c._complementarity_type
                elif _type == 2:
                    raise ValueError('to_standard_form does not generate _type 2 expressions')
            tdata.compl_cuids.append(ComponentUID(complementarity))
            block.reclassify_component_type(complementarity, Block)