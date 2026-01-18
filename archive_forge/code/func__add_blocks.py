import logging
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base import Transformation, Block, SortComponents, TransformationFactory
from pyomo.gdp import Disjunct
from pyomo.network import Arc
from pyomo.network.util import replicate_var
def _add_blocks(self, instance):
    for arc in instance.component_objects(**obj_iter_kwds):
        blk = Block(arc.index_set())
        bname = unique_component_name(arc.parent_block(), '%s_expanded' % arc.local_name)
        arc.parent_block().add_component(bname, blk)
        arc._expanded_block = blk
        if arc.is_indexed():
            for i in arc:
                arc[i]._expanded_block = blk[i]