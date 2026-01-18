from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class TGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for tensors with dynamic type
    """

    def __init__(self, res, rhs1, rhs2):
        """
        :param res: tensor variable that stores the result of the outout
        :param rhs1: tensor or tensor variable
        :param rhs2: tensor or tensor variabke
        """
        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self):
        return f'{self.res} = {self.rhs1}⊔*{self.rhs2}'

    def __eq__(self, other):
        if isinstance(other, TGreatestUpperBound):
            return self.res == other.res and self.rhs1 == other.rhs1 and (self.rhs2 == other.rhs2)
        else:
            return False