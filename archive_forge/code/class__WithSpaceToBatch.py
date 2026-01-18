import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
class _WithSpaceToBatch:
    """Helper class for with_space_to_batch.

  Note that this class assumes that shapes of input and filter passed to
  `__call__` are compatible with `input_shape`, `filter_shape`, and
  `spatial_dims` passed to the constructor.

  Arguments
    input_shape: static shape of input. i.e. input.shape.
    dilation_rate: see `with_space_to_batch`.
    padding: see `with_space_to_batch`.
    build_op: Function that maps (num_spatial_dims, paddings) -> (function that
      maps (input, filter) -> output).
    filter_shape: see `with_space_to_batch`.
    spatial_dims: `see with_space_to_batch`.
    data_format: see `with_space_to_batch`.
    num_batch_dims: (Optional).  Number of batch dims in `input_shape`.
  """

    def __init__(self, input_shape, dilation_rate, padding, build_op, filter_shape=None, spatial_dims=None, data_format=None, num_batch_dims=1):
        """Helper class for _with_space_to_batch."""
        dilation_rate = ops.convert_to_tensor(dilation_rate, dtypes.int32, name='dilation_rate')
        if dilation_rate.shape.ndims not in (None, 1):
            raise ValueError(f'`dilation_rate.shape.rank` must be 1. Received: dilation_rate={dilation_rate} of rank {dilation_rate.shape.rank}')
        if not dilation_rate.shape.is_fully_defined():
            raise ValueError(f'`dilation_rate.shape` must be fully defined. Received: dilation_rate={dilation_rate} with shape {dilation_rate.shape}')
        num_spatial_dims = dilation_rate.shape.dims[0].value
        if data_format is not None and data_format.startswith('NC'):
            starting_spatial_dim = num_batch_dims + 1
        else:
            starting_spatial_dim = num_batch_dims
        if spatial_dims is None:
            spatial_dims = range(starting_spatial_dim, num_spatial_dims + starting_spatial_dim)
        orig_spatial_dims = list(spatial_dims)
        spatial_dims = sorted(set((int(x) for x in orig_spatial_dims)))
        if spatial_dims != orig_spatial_dims or any((x < 1 for x in spatial_dims)):
            raise ValueError(f'`spatial_dims` must be a monotonically increasing sequence of positive integers. Received: spatial_dims={orig_spatial_dims}')
        if data_format is not None and data_format.startswith('NC'):
            expected_input_rank = spatial_dims[-1]
        else:
            expected_input_rank = spatial_dims[-1] + 1
        try:
            input_shape.with_rank_at_least(expected_input_rank)
        except ValueError:
            raise ValueError(f'`input.shape.rank` must be at least {expected_input_rank}. Received: input.shape={input_shape} with rank {input_shape.rank}')
        const_rate = tensor_util.constant_value(dilation_rate)
        rate_or_const_rate = dilation_rate
        if const_rate is not None:
            rate_or_const_rate = const_rate
            if np.any(const_rate < 1):
                raise ValueError(f'`dilation_rate` must be positive. Received: dilation_rate={const_rate}')
            if np.all(const_rate == 1):
                self.call = build_op(num_spatial_dims, padding)
                return
        padding, explicit_paddings = convert_padding(padding)
        if padding == 'SAME':
            if filter_shape is None:
                raise ValueError(f"`filter_shape` must be specified for `padding='SAME'`. Received: filter_shape={filter_shape} and padding={padding}")
            filter_shape = ops.convert_to_tensor(filter_shape, name='filter_shape')
            const_filter_shape = tensor_util.constant_value(filter_shape)
            if const_filter_shape is not None:
                filter_shape = const_filter_shape
                self.base_paddings = _with_space_to_batch_base_paddings(const_filter_shape, num_spatial_dims, rate_or_const_rate)
            else:
                self.num_spatial_dims = num_spatial_dims
                self.rate_or_const_rate = rate_or_const_rate
                self.base_paddings = None
        elif padding == 'VALID':
            self.base_paddings = np.zeros([num_spatial_dims, 2], np.int32)
        elif padding == 'EXPLICIT':
            base_paddings = np.array(explicit_paddings).reshape([num_spatial_dims + 2, 2])
            if data_format is not None and data_format.startswith('NC'):
                self.base_paddings = base_paddings[2:]
            else:
                self.base_paddings = base_paddings[1:-1]
        else:
            raise ValueError(f"`padding` must be one of 'SAME' or 'VALID'. Received: padding={padding}")
        self.input_shape = input_shape
        self.spatial_dims = spatial_dims
        self.dilation_rate = dilation_rate
        self.data_format = data_format
        self.op = build_op(num_spatial_dims, 'VALID')
        self.call = self._with_space_to_batch_call

    def _with_space_to_batch_call(self, inp, filter):
        """Call functionality for with_space_to_batch."""
        input_spatial_shape = None
        input_shape = self.input_shape
        spatial_dims = self.spatial_dims
        if input_shape.ndims is not None:
            input_shape_list = input_shape.as_list()
            input_spatial_shape = [input_shape_list[i] for i in spatial_dims]
        if input_spatial_shape is None or None in input_spatial_shape:
            input_shape_tensor = array_ops.shape(inp)
            input_spatial_shape = array_ops_stack.stack([input_shape_tensor[i] for i in spatial_dims])
        base_paddings = self.base_paddings
        if base_paddings is None:
            filter_shape = array_ops.shape(filter)
            base_paddings = _with_space_to_batch_base_paddings(filter_shape, self.num_spatial_dims, self.rate_or_const_rate)
        paddings, crops = array_ops.required_space_to_batch_paddings(input_shape=input_spatial_shape, base_paddings=base_paddings, block_shape=self.dilation_rate)
        dilation_rate = _with_space_to_batch_adjust(self.dilation_rate, 1, spatial_dims)
        paddings = _with_space_to_batch_adjust(paddings, 0, spatial_dims)
        crops = _with_space_to_batch_adjust(crops, 0, spatial_dims)
        input_converted = array_ops.space_to_batch_nd(input=inp, block_shape=dilation_rate, paddings=paddings)
        result = self.op(input_converted, filter)
        result_converted = array_ops.batch_to_space_nd(input=result, block_shape=dilation_rate, crops=crops)
        if self.data_format is not None and self.data_format.startswith('NC'):
            if not result_converted.shape.dims[1].value and filter is not None:
                output_shape = result_converted.shape.as_list()
                output_shape[1] = filter.shape[-1]
                result_converted.set_shape(output_shape)
        return result_converted

    def __call__(self, inp, filter):
        return self.call(inp, filter)