import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
def Xtest_setitem(self):
    """Check the access to items"""
    try:
        self.model.Y = Set(initialize=[1, 2])
        self.model.Z = Set(initialize=['A', 'C'])
        self.model.A = Set(self.model.Z, self.model.Y, initialize={'A': [1]})
        self.instance = self.model.create_instance()
        tmp = [1, 6, 9]
        self.instance.A['A'] = tmp
        self.instance.A['C'] = tmp
    except:
        self.fail('Problems setting a valid set into a set array')
    try:
        self.instance.A['D'] = tmp
    except KeyError:
        pass
    else:
        self.fail('Problems setting an invalid set into a set array')