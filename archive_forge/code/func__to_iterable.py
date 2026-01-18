from pyomo.dae.flatten import flatten_dae_components
from pyomo.common.modeling import NOTSET
from pyomo.core.base.var import Var
from pyomo.core.base.expression import Expression
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.expr.numeric_expr import value as pyo_value
from pyomo.contrib.mpc.interfaces.load_data import (
from pyomo.contrib.mpc.interfaces.copy_values import copy_values_at_time
from pyomo.contrib.mpc.data.find_nearest_index import find_nearest_index
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.convert import _process_to_dynamic_data
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.modeling.constraints import get_piecewise_constant_constraints
def _to_iterable(item):
    if hasattr(item, '__iter__'):
        if isinstance(item, iterable_scalars):
            yield item
        else:
            for obj in item:
                yield obj
    else:
        yield item