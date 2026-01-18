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
class ObjectGroupParam(pTypes.GroupParameter):

    def __init__(self):
        pTypes.GroupParameter.__init__(self, name='Objects', addText='Add New..', addList=['Clock', 'Grid'])

    def addNew(self, typ):
        if typ == 'Clock':
            self.addChild(ClockParam())
        elif typ == 'Grid':
            self.addChild(GridParam())