import ctypes
import enum
import os
import platform
import sys
import numpy as np
def _get_tensor_details(self, tensor_index, subgraph_index):
    """Gets tensor details.

    Args:
      tensor_index: Tensor index of tensor to query.
      subgraph_index: Index of the subgraph.

    Returns:
      A dictionary containing the following fields of the tensor:
        'name': The tensor name.
        'index': The tensor index in the interpreter.
        'shape': The shape of the tensor.
        'quantization': Deprecated, use 'quantization_parameters'. This field
            only works for per-tensor quantization, whereas
            'quantization_parameters' works in all cases.
        'quantization_parameters': The parameters used to quantize the tensor:
          'scales': List of scales (one if per-tensor quantization)
          'zero_points': List of zero_points (one if per-tensor quantization)
          'quantized_dimension': Specifies the dimension of per-axis
              quantization, in the case of multiple scales/zero_points.

    Raises:
      ValueError: If tensor_index is invalid.
    """
    tensor_index = int(tensor_index)
    subgraph_index = int(subgraph_index)
    tensor_name = self._interpreter.TensorName(tensor_index, subgraph_index)
    tensor_size = self._interpreter.TensorSize(tensor_index, subgraph_index)
    tensor_size_signature = self._interpreter.TensorSizeSignature(tensor_index, subgraph_index)
    tensor_type = self._interpreter.TensorType(tensor_index, subgraph_index)
    tensor_quantization = self._interpreter.TensorQuantization(tensor_index, subgraph_index)
    tensor_quantization_params = self._interpreter.TensorQuantizationParameters(tensor_index, subgraph_index)
    tensor_sparsity_params = self._interpreter.TensorSparsityParameters(tensor_index, subgraph_index)
    if not tensor_type:
        raise ValueError('Could not get tensor details')
    details = {'name': tensor_name, 'index': tensor_index, 'shape': tensor_size, 'shape_signature': tensor_size_signature, 'dtype': tensor_type, 'quantization': tensor_quantization, 'quantization_parameters': {'scales': tensor_quantization_params[0], 'zero_points': tensor_quantization_params[1], 'quantized_dimension': tensor_quantization_params[2]}, 'sparsity_parameters': tensor_sparsity_params}
    return details