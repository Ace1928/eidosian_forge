from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class ApplyBroadcasting(Constraint):

    def __init__(self, res1, res2, input1, input2):
        """
        :param res1: resulting tensor 1
        :param res2: resulting tensor 2
        :param input1: tensor variable 1
        :param input2: tensor variable 2
        """
        self.res1 = res1
        self.res2 = res2
        self.input1 = input1
        self.input2 = input2

    def __eq__(self, other):
        if isinstance(other, ApplyBroadcasting):
            return self.res1 == other.res1 and self.res2 == other.res2 and (self.input1 == other.input1) and (self.input2 == other.input2)
        else:
            return False

    def __repr__(self):
        return f'{self.res1}, {self.res2} = apply-broadcasting({self.input1}, {self.input2})'