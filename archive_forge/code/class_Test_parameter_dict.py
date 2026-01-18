import pickle
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import dill, dill_available as has_dill
from pyomo.core.expr.numvalue import (
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.parameter import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.block import block
class Test_parameter_dict(_TestActiveDictContainerBase, unittest.TestCase):
    _container_type = parameter_dict
    _ctype_factory = lambda self: parameter()