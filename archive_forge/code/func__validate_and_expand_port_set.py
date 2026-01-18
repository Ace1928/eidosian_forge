import logging
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base import Transformation, Block, SortComponents, TransformationFactory
from pyomo.gdp import Disjunct
from pyomo.network import Arc
from pyomo.network.util import replicate_var
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