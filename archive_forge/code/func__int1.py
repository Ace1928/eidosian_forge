import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Set, TransformationFactory, Expression
from pyomo.dae import ContinuousSet, Integral
from pyomo.dae.diffvar import DAE_Error
from pyomo.repn import generate_standard_repn
def _int1(m, t):
    return m.v[t]