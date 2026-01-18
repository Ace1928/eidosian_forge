import logging
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base import Transformation, Block, SortComponents, TransformationFactory
from pyomo.gdp import Disjunct
from pyomo.network import Arc
from pyomo.network.util import replicate_var
def _apply_to(self, instance, **kwds):
    if is_debug_set(logger):
        logger.debug('Calling ArcExpander')
    port_list, known_port_sets, matched_ports = self._collect_ports(instance)
    self._add_blocks(instance)
    for port in port_list:
        ref = known_port_sets[id(matched_ports[port])]
        for k, v in sorted(ref.items()):
            rule, kwds = port._rules[k]
            if v[1] >= 0:
                index_set = v[0].index_set()
            else:
                index_set = UnindexedComponent_set
            rule(port, k, index_set, **kwds)
    for arc in instance.component_objects(**obj_iter_kwds):
        arc.deactivate()