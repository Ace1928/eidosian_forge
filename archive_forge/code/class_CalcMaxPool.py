from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class CalcMaxPool(Constraint):

    def __init__(self, maxpool_result, input_var, kernel, padding, stride, dilation, matching_constraint_vars):
        """
        :param maxpool_result: the result of maxpool
        :param input_var: input to convolution
        :param kernel: kernel tuple
        """
        self.maxpool_result = maxpool_result
        self.input_var = input_var
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self):
        return f'{self.maxpool_result} = calc-maxpool({self.input_var},  {self.kernel}, {self.padding}, {self.stride}, {self.dilation})'

    def __eq__(self, other):
        if isinstance(other, CalcMaxPool):
            return self.maxpool_result == other.maxpool_result and self.input_var == other.input_var and (self.kernel == other.kernel) and (self.padding == other.padding) and (self.stride == other.stride) and (self.dilation == other.dilation) and (self.matching_constraint == other.matching_constraint)
        else:
            return False