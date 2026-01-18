import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.variable import IVariable
from pyomo.core.kernel.constraint import IConstraint
class IJunk(IBlock):
    __slots__ = ()