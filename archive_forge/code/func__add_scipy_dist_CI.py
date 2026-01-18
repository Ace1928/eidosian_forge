import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def _add_scipy_dist_CI(x, y, color, columns, ncells, alpha, dist, theta_star, label=None):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax, columns)
    X, Y = _get_XYgrid(x, y, ncells)
    data_slice = []
    if isinstance(dist, stats._multivariate.multivariate_normal_frozen):
        for var in theta_star.index:
            if var == xvar:
                data_slice.append(X)
            elif var == yvar:
                data_slice.append(Y)
            elif var not in [xvar, yvar]:
                data_slice.append(np.array([[theta_star[var]] * ncells] * ncells))
        data_slice = np.dstack(tuple(data_slice))
    elif isinstance(dist, stats.gaussian_kde):
        for var in theta_star.index:
            if var == xvar:
                data_slice.append(X.ravel())
            elif var == yvar:
                data_slice.append(Y.ravel())
            elif var not in [xvar, yvar]:
                data_slice.append(np.array([theta_star[var]] * ncells * ncells))
        data_slice = np.array(data_slice)
    else:
        return
    Z = dist.pdf(data_slice)
    Z = Z.reshape((ncells, ncells))
    ax.contour(X, Y, Z, levels=[alpha], colors=color)