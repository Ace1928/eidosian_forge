from pyomo.common.collections import ComponentMap
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import interval_to_series, _process_to_dynamic_data
A function to get a penalty expression for specified variables from
    a target that is constant, piecewise constant, or time-varying.

    This function accepts ScalarData, IntervalData, or TimeSeriesData objects,
    or compatible mappings/tuples as the target, and builds the appropriate
    penalty expression for each. Mappings are converted to ScalarData, and
    tuples (of data dict, time list) are unpacked and converted to IntervalData
    or TimeSeriesData depending on the contents of the time list.

    Arguments
    ---------
    variables: List
        List of time-indexed variables to be penalized
    time: Set
        Set of time points at which to construct penalty expressions.
        Also indexes the returned Expression.
    setpoint_data: ScalarData, TimeSeriesData, or IntervalData
        Data structure representing the possibly time-varying or piecewise
        constant setpoint
    weight_data: ScalarData (optional)
        Data structure holding the weights to be applied to each variable
    variable_set: Set (optional)
        Set indexing the provided variables, if one already exists. Also
        indexes the returned Expression.
    tolerance: Float (optional)
        Tolerance for checking inclusion within an interval. Only may be
        provided if IntervalData is provided as the setpoint.
    prefer_left: Bool (optional)
        Flag indicating whether left endpoints of intervals should take
        precedence over right endpoints. Default is False. Only may be
        provided if IntervalData is provided as the setpoint.

    Returns
    -------
    Set, Expression
        Set indexing the list of provided variables and an Expression,
        indexed by this set and the provided time set, containing the
        penalties for each variable at each point in time.

    