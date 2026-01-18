from pyomo.common.collections import ComponentMap
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import interval_to_series, _process_to_dynamic_data
def _get_penalty_expressions_from_time_varying_target(variables, time, setpoint_data, weight_data=None):
    if weight_data is None:
        weight_data = ScalarData(ComponentMap(((var, 1.0) for var in variables)))
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    if not isinstance(setpoint_data, TimeSeriesData):
        setpoint_data = TimeSeriesData(*setpoint_data)
    if list(time) != setpoint_data.get_time_points():
        raise RuntimeError('Mismatch in time points between time set and points in the setpoint data structure')
    for var in variables:
        if not setpoint_data.contains_key(var):
            raise KeyError('Setpoint data does not contain a key for variable %s' % var)
        if not weight_data.contains_key(var):
            raise KeyError('Tracking weight does not contain a key for variable %s' % var)
    cuids = [get_indexed_cuid(var, sets=(time,)) for var in variables]
    weights = [weight_data.get_data_from_key(var) for var in variables]
    setpoints = [setpoint_data.get_data_from_key(var) for var in variables]
    tracking_costs = [{t: get_quadratic_penalty_at_time(var, t, setpoints[j][i], weights[j]) for i, t in enumerate(time)} for j, var in enumerate(variables)]
    return tracking_costs