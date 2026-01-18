from pyomo.version import version_info, __version__
import pyomo.environ
import pyomo.opt
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.boolean_value import BooleanValue
from pyomo.core.expr import (
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.taylor_series import taylor_series_expansion
import pyomo.core.kernel
from pyomo.kernel.util import generate_names, preorder_traversal, pprint
from pyomo.core.kernel.variable import (
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.matrix_constraint import matrix_constraint
import pyomo.core.kernel.conic as conic
from pyomo.core.kernel.parameter import (
from pyomo.core.kernel.expression import (
from pyomo.core.kernel.objective import (
from pyomo.core.kernel.sos import sos, sos1, sos2, sos_tuple, sos_list, sos_dict
from pyomo.core.kernel.suffix import (
from pyomo.core.kernel.block import block, block_tuple, block_list, block_dict
from pyomo.core.kernel.piecewise_library.transforms import piecewise
from pyomo.core.kernel.piecewise_library.transforms_nd import piecewise_nd
from pyomo.core.kernel.set_types import RealSet, IntegerSet, BooleanSet
from pyomo.environ import (
from pyomo.core.kernel.base import _convert_ctype
from pyomo.core.kernel.base import _convert_ctype, _kernel_ctype_backmap
import pyomo.core.kernel.piecewise_library.util as piecewise_util
from pyomo.core.kernel.heterogeneous_container import (
def _block_data_objects(self, **kwds):
    kwds.pop('sort', None)
    active = kwds.get('active', None)
    assert active in (None, True)
    if active is not None and (not self.active):
        return
    yield self
    yield from self.components(ctype=self.ctype, **kwds)