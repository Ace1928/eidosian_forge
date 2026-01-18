from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable
from pyomo.contrib.mpc.data.find_nearest_index import (
def _raise_invalid_cuid(cuid, model):
    raise RuntimeError('Cannot find a component %s on block %s' % (cuid, model))