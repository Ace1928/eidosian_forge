from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import standardize_data_format
from keras.src.backend.common.backend_utils import (
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class CtcLoss(Operation):

    def __init__(self, mask_index):
        super().__init__()
        self.mask_index = mask_index

    def call(self, target, output, target_length, output_length):
        return backend.nn.ctc_loss(target, output, target_length, output_length, self.mask_index)

    def _check_shape_first_dim(self, name1, shape1, name2, shape2):
        if shape1[0] != shape2[0]:
            raise ValueError(f'Arguments `{name1}` and `{name2}` must have the same first dimension. Received shapes: `{shape1}` and `{shape2}`.')

    def compute_output_spec(self, target, output, target_length, output_length):
        self._check_shape_first_dim('target', target.shape, 'output', output.shape)
        self._check_shape_first_dim('target_length', target_length.shape, 'target', target.shape)
        self._check_shape_first_dim('output_length', output_length.shape, 'output', output.shape)
        return KerasTensor((target.shape[0],), dtype=target.dtype)