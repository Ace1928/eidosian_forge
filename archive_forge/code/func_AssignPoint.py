from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
@m.Disjunct(m.Points, m.Centroids)
def AssignPoint(d, i, k):
    m = d.model()
    d.LocalVars = Suffix(direction=Suffix.LOCAL)
    d.LocalVars[d] = [m.t[i, k]]

    def distance1(d):
        return m.t[i, k] >= m.X[i] - m.cluster_center[k]

    def distance2(d):
        return m.t[i, k] >= -(m.X[i] - m.cluster_center[k])
    d.dist1 = Constraint(rule=distance1)
    d.dist2 = Constraint(rule=distance2)
    d.define_distance = Constraint(expr=m.distance[i] == m.t[i, k])