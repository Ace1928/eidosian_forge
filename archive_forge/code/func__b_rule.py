import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def _b_rule(b, id):
    b.X = Var()
    b.PORT = Connector()
    b.PORT.add(b.X)