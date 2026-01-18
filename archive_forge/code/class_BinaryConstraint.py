from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class BinaryConstraint(Constraint):
    """
    Represents all binary operations
    """

    def __init__(self, lhs, rhs, op):
        """
        :param lhs: lhs of the constraint
        :param rhs: rhs of the constraint
        :param op: string representing the operation
        """
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __eq__(self, other):
        if isinstance(other, BinaryConstraint):
            return self.lhs == other.lhs and self.rhs == other.rhs and (self.op == other.op)
        else:
            return False

    def __repr__(self):
        return f'({self.lhs} {self.op} {self.rhs})'