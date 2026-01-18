from collections.abc import MutableMapping
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.find_nearest_index import find_nearest_interval_index
def _process_to_dynamic_data(data, time_set=None):
    """Processes a user's data to convert it to the appropriate type
    of dynamic data

    Mappings are converted to ScalarData, and length-two tuples are converted
    to TimeSeriesData or IntervalData, depending on the contents of the
    second item (the list of time points or intervals).

    Arguments
    ---------
    data: Dict, ComponentMap, or Tuple
        Data to convert to either ScalarData, TimeSeriesData, or
        IntervalData, depending on type.

    Returns
    -------
    ScalarData, TimeSeriesData, or IntervalData

    """
    if isinstance(data, _DynamicDataBase):
        return data
    if isinstance(data, MutableMapping):
        return ScalarData(data, time_set=time_set)
    elif isinstance(data, tuple):
        if len(data) != 2:
            raise TypeError('_process_to_dynamic_data only accepts a mapping or a tuple of length two. Got tuple of length %s' % len(data))
        if not isinstance(data[0], MutableMapping):
            raise TypeError('First entry of data tuple must be instance of MutableMapping,e.g. dict or ComponentMap. Got %s' % type(data[0]))
        elif len(data[1]) == 0:
            raise ValueError('Time sequence provided in data tuple is empty. Cannot infer whether this is a list of points or intervals.')
        elif all((not _is_iterable(item) for item in data[1])):
            return TimeSeriesData(*data)
        elif all((_is_iterable(item) and len(item) == 2 for item in data[1])):
            return IntervalData(*data)
        else:
            raise TypeError('Second entry of data tuple must be a non-empty iterable of scalars (time points) or length-two tuples (intervals). Got %s' % str(data[1]))