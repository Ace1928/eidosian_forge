import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def _get_data_slice(xvar, yvar, columns, data, theta_star):
    search_ranges = {}
    for var in columns:
        if var in [xvar, yvar]:
            search_ranges[var] = data[var].unique()
        else:
            search_ranges[var] = [theta_star[var]]
    data_slice = pd.DataFrame(list(itertools.product(*search_ranges.values())), columns=search_ranges.keys())
    for col in data[columns].columns:
        cv = data[col].std() / data[col].mean()
        if cv < 1e-08:
            temp = data.copy()
            if cv == 0:
                temp[col] = temp[col] + data[col].mean() / 10
            else:
                temp[col] = temp[col] + data[col].std()
            data = pd.concat([data, temp], ignore_index=True)
    data_slice['obj'] = scipy.interpolate.griddata(np.array(data[columns]), np.array(data[['obj']]), np.array(data_slice[columns]), method='linear', rescale=True)
    X = data_slice[xvar]
    Y = data_slice[yvar]
    Z = data_slice['obj']
    return (X, Y, Z)