from fontTools.varLib.models import VariationModel, supportScalar
from fontTools.designspaceLib import DesignSpaceDocument
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d
from itertools import cycle
import math
import logging
import sys
def _plotModelFromMasters2D(model, masterValues, fig, **kwargs):
    assert len(model.axisOrder) == 1
    axis = model.axisOrder[0]
    axis_min = min((loc.get(axis, 0) for loc in model.locations))
    axis_max = max((loc.get(axis, 0) for loc in model.locations))
    import numpy as np
    X = np.arange(axis_min, axis_max, (axis_max - axis_min) / 100)
    Y = []
    for x in X:
        loc = {axis: x}
        v = model.interpolateFromMasters(loc, masterValues)
        Y.append(v)
    subplot = fig.add_subplot(111)
    subplot.plot(X, Y, '-', **kwargs)