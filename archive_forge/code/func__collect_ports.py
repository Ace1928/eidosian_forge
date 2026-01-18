import logging
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base import Transformation, Block, SortComponents, TransformationFactory
from pyomo.gdp import Disjunct
from pyomo.network import Arc
from pyomo.network.util import replicate_var
def _collect_ports(self, instance):
    port_list = []
    groupID = 0
    port_groups = dict()
    matched_ports = ComponentMap()
    for arc in instance.component_data_objects(**obj_iter_kwds):
        ref = None
        for p in arc.ports:
            if p in matched_ports:
                if ref is None:
                    ref = matched_ports[p]
                elif ref is not matched_ports[p]:
                    src = matched_ports[p]
                    if len(ref) < len(src):
                        ref, src = (src, ref)
                    ref.update(src)
                    for i in src:
                        matched_ports[i] = ref
                    del port_groups[id(src)]
            else:
                port_list.append(p)
                if ref is None:
                    ref = ComponentSet()
                    port_groups[id(ref)] = (groupID, ref)
                    groupID += 1
                ref.add(p)
                matched_ports[p] = ref
    known_port_sets = {}
    for groupID, port_set in sorted(port_groups.values()):
        known_port_sets[id(port_set)] = self._validate_and_expand_port_set(port_set)
    return (port_list, known_port_sets, matched_ports)