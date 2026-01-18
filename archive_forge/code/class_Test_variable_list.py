import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import (
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.variable import (
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import RealSet, IntegerSet, BooleanSet
from pyomo.core.base.set import (
class Test_variable_list(_TestActiveListContainerBase, unittest.TestCase):
    _container_type = variable_list
    _ctype_factory = lambda self: variable()