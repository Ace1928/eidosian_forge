import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def _con2(_b, i, j, ti):
    return _b.v2[i, j, ti] - _b.v3[ti, i, j] + _b.p1['A', ti]