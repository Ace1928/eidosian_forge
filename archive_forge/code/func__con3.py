import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def _con3(m, i, ti, ti2, j, k):
    return m.v1[i, ti] - m.v3[ti2, j, k] * m.p1[i, ti]