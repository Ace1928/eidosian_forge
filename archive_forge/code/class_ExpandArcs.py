import logging
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base import Transformation, Block, SortComponents, TransformationFactory
from pyomo.gdp import Disjunct
from pyomo.network import Arc
from pyomo.network.util import replicate_var
@TransformationFactory.register('network.expand_arcs', doc='Expand all Arcs in the model to simple constraints')
class ExpandArcs(Transformation):

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

    def _validate_and_expand_port_set(self, ports):
        ref = {}
        for p in ports:
            for k, v in p.vars.items():
                if k in ref:
                    continue
                if v is None:
                    continue
                _len = -1 if not v.is_indexed() else len(v)
                ref[k] = (v, _len, p, p.rule_for(k))
        if not ref:
            logger.warning('Cannot identify a reference port: no ports in the port set have assigned variables:\n\t(%s)' % ', '.join(sorted((p.name for p in ports.values()))))
            return ref
        empty_or_partial = []
        for p in ports:
            p_is_partial = False
            if not p.vars:
                empty_or_partial.append(p)
                continue
            for k, v in ref.items():
                if k not in p.vars:
                    raise ValueError("Port mismatch: Port '%s' missing variable '%s' (appearing in reference port '%s')" % (p.name, k, v[2].name))
                _v = p.vars[k]
                if _v is None:
                    if not p_is_partial:
                        empty_or_partial.append(p)
                        p_is_partial = True
                    continue
                _len = -1 if not _v.is_indexed() else len(_v)
                if (_len >= 0) ^ (v[1] >= 0):
                    raise ValueError("Port mismatch: Port variable '%s' mixing indexed and non-indexed targets on ports '%s' and '%s'" % (k, v[2].name, p.name))
                if _len >= 0 and _len != v[1]:
                    raise ValueError("Port mismatch: Port variable '%s' index mismatch (%s elements in reference port '%s', but %s elements in port '%s')" % (k, v[1], v[2].name, _len, p.name))
                if v[1] >= 0 and len(v[0].index_set() ^ _v.index_set()):
                    raise ValueError("Port mismatch: Port variable '%s' has mismatched indices on ports '%s' and '%s'" % (k, v[2].name, p.name))
                if p.rule_for(k) is not v[3]:
                    raise ValueError("Port mismatch: Port variable '%s' has different rules on ports '%s' and '%s'" % (k, v[2].name, p.name))
        sorted_refs = sorted(ref.items())
        if len(empty_or_partial) > 1:
            empty_or_partial.sort(key=lambda x: x.getname(fully_qualified=True))
        for p in empty_or_partial:
            block = p.parent_block()
            for k, v in sorted_refs:
                if k in p.vars and p.vars[k] is not None:
                    continue
                vname = unique_component_name(block, '%s_auto_%s' % (p.getname(fully_qualified=True), k))
                new_var = replicate_var(v[0], vname, block)
                p.add(new_var, k, rule=v[3])
        return ref

    def _add_blocks(self, instance):
        for arc in instance.component_objects(**obj_iter_kwds):
            blk = Block(arc.index_set())
            bname = unique_component_name(arc.parent_block(), '%s_expanded' % arc.local_name)
            arc.parent_block().add_component(bname, blk)
            arc._expanded_block = blk
            if arc.is_indexed():
                for i in arc:
                    arc[i]._expanded_block = blk[i]