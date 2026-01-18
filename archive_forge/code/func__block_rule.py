import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def _block_rule(b, t, s):
    m = b.model()

    def _init(m, j):
        return j * 2
    b.p1 = Param(m.t, default=_init)
    b.v1 = Var(m.t, initialize=5)