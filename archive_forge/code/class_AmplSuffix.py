import itertools
import logging
import operator
import os
import time
from math import isclose
from pyomo.common.fileutils import find_library
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat, AbstractProblemWriter, WriterFactory
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.base import (
import pyomo.core.base.suffix
from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.expression import IIdentityExpression
from pyomo.core.kernel.variable import IVariable
class AmplSuffix(object):

    def __init__(self, name):
        self.name = name
        self.ids = []
        self.vals = []

    def add(self, idx, val):
        if idx in self.ids:
            raise RuntimeError('The NL file format does not support multiple nonzero values for a single component and suffix. \nSuffix Name:  %s\nComponent ID: %s\n' % (self.name, idx))
        else:
            self.ids.append(idx)
            self.vals.append(val)

    def genfilelines(self):
        base_line = '{0} {1}\n'
        return [base_line.format(idx, val) for idx, val in zip(self.ids, self.vals) if val != 0]

    def is_empty(self):
        return not bool(len(self.ids))