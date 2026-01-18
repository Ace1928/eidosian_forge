import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.symbol_map import symbol_map_from_instance
class FOO(object):

    def __call__(self, *args):
        return 'x'