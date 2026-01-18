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
def _add_equality_constraint(arc, name, index_set):
    eblock = arc.expanded_block
    cname = name + '_equality'
    if eblock.component(cname) is not None:
        return
    port1, port2 = arc.ports
    if index_set is not UnindexedComponent_set:

        def rule(m, *args):
            return port1.vars[name][args] == port2.vars[name][args]
    else:

        def rule(m):
            return port1.vars[name] == port2.vars[name]
    con = Constraint(index_set, rule=rule)
    eblock.add_component(cname, con)