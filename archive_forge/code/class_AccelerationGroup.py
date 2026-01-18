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
class AccelerationGroup(pTypes.GroupParameter):

    def __init__(self, **kwds):
        defs = dict(name='Acceleration', addText='Add Command..')
        pTypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)

    def addNew(self):
        nextTime = 0.0
        if self.hasChildren():
            nextTime = self.children()[-1]['Proper Time'] + 1
        self.addChild(Parameter.create(name='Command', autoIncrementName=True, type=None, renamable=True, removable=True, children=[dict(name='Proper Time', type='float', value=nextTime), dict(name='Acceleration', type='float', value=0.0, step=0.1)]))

    def generate(self):
        prog = []
        for cmd in self:
            prog.append((cmd['Proper Time'], cmd['Acceleration']))
        return prog