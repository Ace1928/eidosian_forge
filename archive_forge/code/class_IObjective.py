from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.kernel.base import _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.kernel.expression import IExpression
class IObjective(IExpression):
    """
    The interface for optimization objectives.
    """
    __slots__ = ()
    sense = _abstract_readwrite_property(doc='The optimization direction for the objective (minimize or maximize)')

    def is_minimizing(self):
        return self.sense == minimize