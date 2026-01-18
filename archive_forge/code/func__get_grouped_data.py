import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def _get_grouped_data(data1, data2, normalize, group_names):
    if normalize:
        data_median = data1.median()
        data_std = data1.std()
        data1 = (data1 - data_median) / data_std
        data2 = (data2 - data_median) / data_std
    data = pd.concat({group_names[0]: data1, group_names[1]: data2})
    data.reset_index(level=0, inplace=True)
    data.rename(columns={'level_0': 'set'}, inplace=True)
    data = data.melt(id_vars='set', value_vars=data1.columns, var_name='columns')
    return data