import pickle
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.heterogeneous_container import IHeterogeneousContainer
from pyomo.core.kernel.block import IBlock, block, block_dict, block_list
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.piecewise_library.transforms import (
import pyomo.core.kernel.piecewise_library.transforms as transforms
from pyomo.core.kernel.piecewise_library.transforms_nd import (
import pyomo.core.kernel.piecewise_library.transforms_nd as transforms_nd
import pyomo.core.kernel.piecewise_library.util as util
@unittest.skipUnless(util.numpy_available and util.scipy_available, 'Numpy or Scipy is not available')
class Test_piecewise_nd_list(_TestActiveListContainerBase, unittest.TestCase):
    _container_type = block_list
    _ctype_factory = lambda self: transforms_nd.piecewise_nd(_test_tri, _test_values)