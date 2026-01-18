import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def _add_rectangle_CI(x, y, color, columns, lower_bound, upper_bound, label=None):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax, columns)
    xmin = lower_bound[xvar]
    ymin = lower_bound[yvar]
    xmax = upper_bound[xvar]
    ymax = upper_bound[yvar]
    ax.plot([xmin, xmax], [ymin, ymin], color=color)
    ax.plot([xmax, xmax], [ymin, ymax], color=color)
    ax.plot([xmax, xmin], [ymax, ymax], color=color)
    ax.plot([xmin, xmin], [ymax, ymin], color=color)