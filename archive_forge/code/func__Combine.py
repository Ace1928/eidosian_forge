import logging, sys
from weakref import ref as weakref_ref
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.deprecation import RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr import identify_variables
from pyomo.core.base.label import alphanum_label_from_name
from pyomo.network.util import create_var, tighten_var_domain
@staticmethod
def _Combine(port, name, index_set):
    port_parent = port.parent_block()
    var = port.vars[name]
    in_vars = []
    sources = port.sources(active=True)
    if not len(sources):
        return in_vars
    if len(sources) == 1 and len(sources[0].source.dests(active=True)) == 1:
        arc = sources[0]
        Port._add_equality_constraint(arc, name, index_set)
        return in_vars
    for arc in sources:
        eblock = arc.expanded_block
        evar = Port._create_evar(port.vars[name], name, eblock, index_set)
        in_vars.append(evar)
    if len(sources) == 1:
        tighten_var_domain(port.vars[name], in_vars[0], index_set)
    cname = unique_component_name(port_parent, '%s_%s_insum' % (alphanum_label_from_name(port.local_name), name))
    if index_set is not UnindexedComponent_set:

        def rule(m, *args):
            return sum((evar[args] for evar in in_vars)) == var[args]
    else:

        def rule(m):
            return sum((evar for evar in in_vars)) == var
    con = Constraint(index_set, rule=rule)
    port_parent.add_component(cname, con)
    return in_vars