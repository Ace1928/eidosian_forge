import contextlib
import traceback
import weakref
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export
class _GraphTensorArrayV2:
    """Graph-mode implementation of TensorArray backed by TensorLists.

  The backing tensor of this TensorArray is a TensorList variant tensor which is
  stored in the `flow`. The `handle` is always none here. The reason we use the
  `flow` field and not the `handle` field is to ensure backwards compatibility
  with legacy control flow.
  """

    def __init__(self, dtype, size=None, dynamic_size=None, clear_after_read=None, tensor_array_name=None, handle=None, flow=None, infer_shape=True, element_shape=None, colocate_with_first_write_call=True, name=None):
        """Constructs a graph mode TensorArray.

    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if flow is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: (optional) unused. Not supported in TensorLists.
      tensor_array_name: (optional) unused.
      handle: (optional) Must always be None.
      flow: (optional) A variant `Tensor` scalar for a TensorList.
      infer_shape: (optional, default: True) If True, shape inference is
        enabled.  In this case, all elements must have the same shape.
      element_shape: (optional, default: None) A `TensorShape` object specifying
        the shape constraints of each of the elements of the TensorArray. Need
        not be fully defined.
      colocate_with_first_write_call: (optional). unused.
      name: (optional) A name for the operation.

    Raises:
      ValueError: if both handle and tensor_array_name are provided.
      TypeError: if handle is provided but is not a Tensor.
    """
        assert handle is None
        del handle
        del clear_after_read
        del tensor_array_name
        del colocate_with_first_write_call
        self._dynamic_size = dynamic_size
        self._size = size
        if flow is not None and (not isinstance(flow, tensor_lib.Tensor) or flow.dtype != dtypes.variant):
            raise TypeError(f'Expected `flow` to be a variant tensor, but received `{flow.dtype}` instead.')
        if flow is None and size is None:
            raise ValueError('Argument `size` must be provided if argument `flow` is not provided.')
        if flow is not None and size is not None:
            raise ValueError('Cannot provide both `flow` and `size` arguments at the same time.')
        if flow is not None and element_shape is not None:
            raise ValueError('Cannot provide both `flow` and `element_shape` argumentsat the same time.')
        self._dtype = dtypes.as_dtype(dtype).base_dtype
        self._element_shape = [tensor_shape.as_shape(element_shape)]
        self._infer_shape = infer_shape
        with ops.name_scope(name, 'TensorArrayV2', [size, flow]) as scope:
            if flow is None:
                self._flow = list_ops.tensor_list_reserve(element_shape=element_shape, num_elements=size, element_dtype=dtype, name=scope)
            else:
                self._flow = flow
        self._colocate_with_first_write_call = None
        self._colocate_with = None

    @property
    def flow(self):
        return self._flow

    @property
    def dtype(self):
        return self._dtype

    @property
    def element_shape(self):
        return self._element_shape[0]

    @property
    def handle(self):
        return None

    def _check_element_shape(self, shape):
        """Changes the element shape of the array given a shape to merge with.

    Args:
      shape: A `TensorShape` object to merge with.

    Raises:
      ValueError: if the provided shape is incompatible with the current
          element shape of the `TensorArray`.
    """
        if not shape.is_compatible_with(self.element_shape):
            raise ValueError('Inconsistent shapes: saw %s but expected %s ' % (shape, self.element_shape))
        if self._infer_shape:
            self._element_shape[0] = self.element_shape.merge_with(shape)

    def identity(self):
        """See TensorArray."""
        flow = array_ops.identity(self._flow)
        return build_ta_with_new_flow(self, flow)

    def grad(self, source, flow=None, name=None):
        """Not supported."""
        raise NotImplementedError()

    def read(self, index, name=None):
        """See TensorArray."""
        with ops.name_scope(name, 'TensorArrayV2Read', [self._flow, index]):
            value = list_ops.tensor_list_get_item(input_handle=self._flow, index=index, element_dtype=self._dtype, element_shape=self.element_shape, name=name)
            return value

    def write(self, index, value, name=None):
        """See TensorArray."""
        with ops.name_scope(name, 'TensorArrayV2Write', [self._flow, index, value]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            self._check_element_shape(value.shape)
            flow_out = list_ops.tensor_list_set_item(input_handle=self._flow, index=index, item=value, resize_if_index_out_of_bounds=self._dynamic_size, name=name)
            return build_ta_with_new_flow(self, flow_out)

    def stack(self, name=None):
        """See TensorArray."""
        with ops.name_scope(name, 'TensorArrayV2Stack', [self._flow]):
            if not self._dynamic_size and self._size is not None:
                ta_size = tensor_util.constant_value(self._size)
            else:
                ta_size = -1
            value = list_ops.tensor_list_stack(input_handle=self._flow, element_dtype=self._dtype, num_elements=ta_size, element_shape=self.element_shape)
            return value

    def gather(self, indices, name=None):
        """See TensorArray."""
        value = list_ops.tensor_list_gather(input_handle=self._flow, indices=indices, element_dtype=self._dtype, element_shape=self.element_shape, name=name)
        return value

    def concat(self, name=None):
        """See TensorArray."""
        if self.element_shape:
            element_shape = [None] + self.element_shape.dims[1:]
        else:
            element_shape = None
        value = list_ops.tensor_list_concat(input_handle=self._flow, element_dtype=self._dtype, element_shape=element_shape, name=name)
        return value

    @tf_should_use.should_use_result
    def unstack(self, value, name=None):
        """See TensorArray."""
        with ops.name_scope(name, 'TensorArrayUnstack', [self._flow, value]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            self._check_element_shape(value.shape[1:])
            flow_out = list_ops.tensor_list_from_tensor(tensor=value, element_shape=value.shape[1:])
            return build_ta_with_new_flow(self, flow_out)

    @tf_should_use.should_use_result
    def scatter(self, indices, value, name=None):
        """See TensorArray."""
        with ops.name_scope(name, 'TensorArrayScatter', [self._flow, value, indices]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            self._check_element_shape(value.shape[1:])
            flow_out = list_ops.tensor_list_scatter(tensor=value, indices=indices, element_shape=self.element_shape, input_handle=self._flow)
            return build_ta_with_new_flow(self, flow_out)

    @tf_should_use.should_use_result
    def split(self, value, lengths, name=None):
        """See TensorArray."""
        with ops.name_scope(name, 'TensorArraySplit', [self._flow, value, lengths]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            lengths_64 = math_ops.cast(lengths, dtypes.int64)
            if not context.executing_eagerly():
                clengths = tensor_util.constant_value(lengths_64)
                if value.shape.dims is not None and clengths is not None:
                    if clengths.shape and clengths.max() == clengths.min():
                        self._check_element_shape(tensor_shape.TensorShape([clengths[0]]).concatenate(value.shape[1:]))
            flow_out = list_ops.tensor_list_split(tensor=value, lengths=lengths_64, element_shape=self.element_shape, name=name)
            return build_ta_with_new_flow(self, flow_out)

    def size(self, name=None):
        """See TensorArray."""
        if not self._dynamic_size and self._size is not None:
            return ops.convert_to_tensor(self._size, dtype=dtypes.int32)
        else:
            return list_ops.tensor_list_length(input_handle=self._flow, name=name)

    def close(self, name=None):
        """See TensorArray."""
        return gen_control_flow_ops.no_op(name=name)