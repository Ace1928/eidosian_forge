import pickle
import pyomo.common.unittest as unittest
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.sos import ISOS, sos, sos1, sos2, sos_dict, sos_tuple, sos_list
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression, data_expression
class Test_sos_list(_TestActiveListContainerBase, unittest.TestCase):
    _container_type = sos_list
    _ctype_factory = lambda self: sos([variable()])