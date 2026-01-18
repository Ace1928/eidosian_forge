import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def _get_XYgrid(x, y, ncells):
    xlin = np.linspace(min(x) - abs(max(x) - min(x)) / 2, max(x) + abs(max(x) - min(x)) / 2, ncells)
    ylin = np.linspace(min(y) - abs(max(y) - min(y)) / 2, max(y) + abs(max(y) - min(y)) / 2, ncells)
    X, Y = np.meshgrid(xlin, ylin)
    return (X, Y)