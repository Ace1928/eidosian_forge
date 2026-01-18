import collections.abc
import pickle
import pyomo.common.unittest as unittest
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.suffix import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.constraint import constraint, constraint_list
from pyomo.core.kernel.block import block, block_dict
class Test_suffix_dict(_TestActiveDictContainerBase, unittest.TestCase):
    _container_type = suffix_dict
    _ctype_factory = lambda self: suffix()