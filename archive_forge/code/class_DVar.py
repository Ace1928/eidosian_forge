from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class DVar:
    """
    Dimension variable
    """

    def __init__(self, c):
        """
        :param c: character or number
        """
        self.c = c

    def __repr__(self):
        return f'DV({self.c})'

    def __eq__(self, other):
        if isinstance(other, DVar):
            return self.c == other.c
        else:
            return False