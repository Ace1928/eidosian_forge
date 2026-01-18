import functools
import pickle
import platform
import sys
import types
import pyomo.common.unittest as unittest
from pyomo.common.config import ConfigValue, ConfigList, ConfigDict
from pyomo.common.dependencies import (
from pyomo.core.base.util import flatten_tuple
from pyomo.core.base.initializer import (
from pyomo.environ import ConcreteModel, Var
class InitIndexed(object):

    def __init__(self, val):
        self.val = val

    def __call__(self, m, i):
        return self.val + i