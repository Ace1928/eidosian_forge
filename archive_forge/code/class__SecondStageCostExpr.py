import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
class _SecondStageCostExpr(object):
    """
    Class to pass objective expression into the Pyomo model
    """

    def __init__(self, ssc_function, data):
        self._ssc_function = ssc_function
        self._data = data

    def __call__(self, model):
        return self._ssc_function(model, self._data)