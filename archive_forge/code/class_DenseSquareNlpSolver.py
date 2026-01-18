from collections import namedtuple
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigBlock
from pyomo.util.subsystems import create_subsystem_block
class DenseSquareNlpSolver(SquareNlpSolverBase):
    """A square NLP solver that uses a dense Jacobian"""

    def evaluate_jacobian(self, x0):
        sparse_jac = super().evaluate_jacobian(x0)
        dense_jac = sparse_jac.toarray()
        return dense_jac