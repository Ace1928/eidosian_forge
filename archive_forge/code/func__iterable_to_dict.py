from pyomo.network.port import Port
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from weakref import ref as weakref_ref
import logging, sys
def _iterable_to_dict(vals, directed, name):
    if type(vals) is not dict:
        try:
            ports = tuple(vals)
        except TypeError:
            ports = None
        if ports is None or len(ports) != 2:
            raise ValueError("Value for arc '%s' is not either a dict or a two-member iterable." % name)
        if directed:
            source, destination = ports
            ports = None
        else:
            source = destination = None
        vals = dict(source=source, destination=destination, ports=ports, directed=directed)
    elif 'directed' not in vals:
        vals['directed'] = directed
    return vals