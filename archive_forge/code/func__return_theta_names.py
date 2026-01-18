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
def _return_theta_names(self):
    """
        Return list of fitted model parameter names
        """
    if hasattr(self, 'theta_names_updated'):
        return self.theta_names_updated
    else:
        return self.theta_names