from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def _get_penalty_expressions_at_time(variables, t, target_data, weight_data=None, time_set=None):
    """A private helper function to process data and construct penalty
    expressions

    """
    if weight_data is None:
        weight_data = ScalarData(ComponentMap(((var, 1.0) for var in variables)))
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data, time_set=time_set)
    if not isinstance(target_data, ScalarData):
        target_data = ScalarData(target_data, time_set=time_set)
    for var in variables:
        if not target_data.contains_key(var):
            raise KeyError('Target data does not contain a key for variable %s' % var.name)
        if not weight_data.contains_key(var):
            raise KeyError('Penalty weight data does not contain a key for variable %s' % var.name)
    penalties = [_get_quadratic_penalty_at_time(var, t, target_data.get_data_from_key(var), weight_data.get_data_from_key(var)) for var in variables]
    return penalties