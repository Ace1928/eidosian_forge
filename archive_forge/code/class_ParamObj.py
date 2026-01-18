import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
class ParamObj(object):

    def __init__(self):
        self.__params = {}

    def __setitem__(self, item, val):
        self.setParam(item, val)

    def setParam(self, param, val):
        self.setParams(**{param: val})

    def setParams(self, **params):
        """Set parameters for this optic. This is a good function to override for subclasses."""
        self.__params.update(params)
        self.paramStateChanged()

    def paramStateChanged(self):
        pass

    def __getitem__(self, item):
        return self.getParam(item)

    def __len__(self):
        return 0

    def getParam(self, param):
        return self.__params[param]