from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class BinConstraintT(BinaryConstraint):
    """
    Binary constraints about tensors
    """

    def __init__(self, lhs, rhs, op):
        assert (isinstance(lhs, (TVar, TensorType, int)) or lhs == Dyn) and (isinstance(rhs, (TVar, TensorType, int)) or rhs == Dyn)
        super().__init__(lhs, rhs, op)

    def __eq__(self, other):
        return super().__eq__(other)