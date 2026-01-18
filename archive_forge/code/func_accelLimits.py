import collections
import os
import sys
from time import perf_counter
import numpy as np
import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
def accelLimits(self):
    if len(self.prog) == 0:
        return (-np.inf, np.inf)
    t = self.pt
    ind = -1
    for i, v in enumerate(self.prog):
        t1, f = v
        if t >= t1:
            ind = i
    if ind == -1:
        return (-np.inf, self.prog[0][0])
    elif ind == len(self.prog) - 1:
        return (self.prog[-1][0], np.inf)
    else:
        return (self.prog[ind][0], self.prog[ind + 1][0])