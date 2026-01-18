import logging
from pyomo.core.base import Transformation, TransformationFactory, Block, SortComponents
from pyomo.mpec.complementarity import Complementarity
from pyomo.gdp import Disjunct
@TransformationFactory.register('mpec.standard_form', doc='Standard reformulation of complementarity condition')
class MPEC3_Transformation(Transformation):

    def __init__(self):
        super(MPEC3_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        for complementarity in instance.component_objects(Complementarity, active=True, descend_into=(Block, Disjunct), sort=SortComponents.deterministic):
            block = complementarity.parent_block()
            for index in sorted(complementarity.keys()):
                _data = complementarity[index]
                if not _data.active:
                    continue
                _data.to_standard_form()
            block.reclassify_component_type(complementarity, Block)