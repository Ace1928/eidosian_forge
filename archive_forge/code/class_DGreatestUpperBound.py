from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class DGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for dimensions
    """

    def __init__(self, res, rhs1, rhs2):
        """
        :param res: Dimension variable to store the result
        :param rhs1: dimension variable 1
        :param rhs2: dimension variable 2
        """
        assert is_dim(res)
        assert is_dim(rhs1)
        assert is_dim(rhs2)
        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self):
        return f'{self.res} = {self.rhs1}⊔{self.rhs2}'

    def __eq__(self, other):
        if isinstance(other, DGreatestUpperBound):
            return self.res == other.res and self.rhs1 == other.rhs1 and (self.rhs2 == other.rhs2)
        else:
            return False