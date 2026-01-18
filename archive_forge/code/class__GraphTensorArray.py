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
class _GraphTensorArray:
    """Graph-mode implementation of TensorArray."""

    def __init__(self, dtype, size=None, dynamic_size=None, clear_after_read=None, tensor_array_name=None, handle=None, flow=None, infer_shape=True, element_shape=None, colocate_with_first_write_call=True, name=None):
        """Constructs a graph mode TensorArray.

    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if handle is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: Boolean (optional, default: True).  If True, clear
        TensorArray values after reading them.  This disables read-many
        semantics, but allows early release of memory.
      tensor_array_name: (optional) Python string: the name of the TensorArray.
        This is used when creating the TensorArray handle.  If this value is
        set, handle should be None.
      handle: (optional) A `Tensor` handle to an existing TensorArray.  If this
        is set, tensor_array_name should be None. Only supported in graph mode.
      flow: (optional) A float `Tensor` scalar coming from an existing
        `TensorArray.flow`. Only supported in graph mode.
      infer_shape: (optional, default: True) If True, shape inference is
        enabled.  In this case, all elements must have the same shape.
      element_shape: (optional, default: None) A `TensorShape` object specifying
        the shape constraints of each of the elements of the TensorArray. Need
        not be fully defined.
      colocate_with_first_write_call: If `True`, the TensorArray will be
        colocated on the same device as the Tensor used on its first write
        (write operations include `write`, `unstack`, and `split`).  If `False`,
        the TensorArray will be placed on the device determined by the device
        context available during its initialization.
      name: A name for the operation (optional).

    Raises:
      ValueError: if both handle and tensor_array_name are provided.
      TypeError: if handle is provided but is not a Tensor.
    """
        if handle is not None and tensor_array_name:
            raise ValueError('Cannot provide both `handle` and `tensor_array_name` arguments at the same time.')
        if handle is not None and (not isinstance(handle, tensor_lib.Tensor)):
            raise TypeError(f'Expected `handle` to be a Tensor, but got `{handle}` of type `{type(handle)}` instead.')
        if handle is None and size is None:
            raise ValueError('Argument `size` must be provided if handle is not provided.')
        if handle is not None and size is not None:
            raise ValueError('Cannot provide both a `handle` and `size` arguments at the same time.')
        if handle is not None and element_shape is not None:
            raise ValueError('Cannot provide both `handle` and `element_shape` arguments at the same time.')
        if handle is not None and dynamic_size is not None:
            raise ValueError('Cannot provide both `handle` and `dynamic_size` arguments at the same time.')
        if handle is not None and clear_after_read is not None:
            raise ValueError('Cannot provide both `handle` and `clear_after_read` arguments at the same time.')
        if clear_after_read is None:
            clear_after_read = True
        self._dynamic_size = dynamic_size or False
        self._dtype = dtypes.as_dtype(dtype).base_dtype
        self._colocate_with_first_write_call = colocate_with_first_write_call
        if colocate_with_first_write_call:
            self._colocate_with = []
        else:
            self._colocate_with = None
        self._element_shape = [tensor_shape.as_shape(element_shape)]
        self._infer_shape = infer_shape
        self._size = size
        with ops.name_scope(name, 'TensorArray', [handle, size, flow]) as scope:
            if handle is not None:
                self._handle = handle
                if flow is None:
                    raise ValueError('flow must not be None if handle is not None.')
                self._flow = flow
            else:

                def create():
                    """Create the TensorArray op."""
                    return gen_data_flow_ops.tensor_array_v3(dtype=dtype, size=size, element_shape=element_shape, identical_element_shapes=infer_shape, dynamic_size=self._dynamic_size, clear_after_read=clear_after_read, tensor_array_name=tensor_array_name, name=scope)
                if colocate_with_first_write_call:
                    with ops.device(None), ops.colocate_with(None, ignore_existing=True):
                        self._handle, self._flow = create()
                else:
                    self._handle, self._flow = create()

    @property
    def flow(self):
        return self._flow

    @property
    def dtype(self):
        return self._dtype

    @property
    def handle(self):
        return self._handle

    @property
    def element_shape(self):
        return self._element_shape[0]

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

    @contextlib.contextmanager
    def _maybe_colocate_with(self, value):
        """Colocate operations with an internal colocation group or `value`.

    Args:
      value: `Tensor`, the tensor to try to colocate with.

    Yields:
      Does not yield anything, but the new context is a colocation context.

    If no internal colocation group is set, colocate with `value` and set
    the internal colocation group to be value.
    """
        if not self._colocate_with_first_write_call:
            yield
        else:
            if not self._colocate_with:
                self._colocate_with.append(value)
            with ops.colocate_with(self._colocate_with[0]):
                yield

    def identity(self):
        """See TensorArray."""
        flow = array_ops.identity(self._flow)
        return build_ta_with_new_flow(self, flow)

    def grad(self, source, flow=None, name=None):
        """See TensorArray."""
        if flow is None:
            flow = self.flow
        with ops.name_scope(name, 'TensorArrayGrad', [self._handle]):
            with ops.colocate_with(self._handle):
                g_handle, unused_flow = gen_data_flow_ops.tensor_array_grad_v3(handle=self._handle, source=source, flow_in=flow, name=name)
                with ops.control_dependencies([g_handle]):
                    flow = array_ops.identity(flow, name='gradient_flow')
                g = TensorArray(dtype=self._dtype, handle=g_handle, flow=flow, infer_shape=self._infer_shape, colocate_with_first_write_call=False)
                g._implementation._element_shape = self._element_shape
                return g

    def read(self, index, name=None):
        """See TensorArray."""
        value = gen_data_flow_ops.tensor_array_read_v3(handle=self._handle, index=index, flow_in=self._flow, dtype=self._dtype, name=name)
        if self._element_shape:
            value.set_shape(self._element_shape[0].dims)
        return value

    def write(self, index, value, name=None):
        """See TensorArray."""
        with ops.name_scope(name, 'TensorArrayWrite', [self._handle, index, value]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            self._check_element_shape(value.shape)
            with self._maybe_colocate_with(value):
                flow_out = gen_data_flow_ops.tensor_array_write_v3(handle=self._handle, index=index, value=value, flow_in=self._flow, name=name)
            return build_ta_with_new_flow(self, flow_out)

    def stack(self, name=None):
        """See TensorArray."""
        with ops.colocate_with(self._handle):
            with ops.name_scope(name, 'TensorArrayStack', [self._handle]):
                value = self.gather(math_ops.range(0, self.size()), name=name)
                if self.element_shape and (not self._dynamic_size) and (self._size is not None):
                    value.set_shape([tensor_util.constant_value(self._size)] + self.element_shape.dims)
                return value

    def gather(self, indices, name=None):
        """See TensorArray."""
        if self._element_shape:
            element_shape = self._element_shape[0]
        else:
            element_shape = tensor_shape.unknown_shape(None)
        value = gen_data_flow_ops.tensor_array_gather_v3(handle=self._handle, indices=indices, flow_in=self._flow, dtype=self._dtype, name=name, element_shape=element_shape)
        if self.element_shape:
            value.set_shape([None] + self.element_shape.dims)
        return value

    def concat(self, name=None):
        """See TensorArray."""
        value, _ = gen_data_flow_ops.tensor_array_concat_v3(handle=self._handle, flow_in=self._flow, dtype=self._dtype, name=name, element_shape_except0=self.element_shape[1:])
        if self.element_shape:
            dim0 = None
            if self._infer_shape:
                size = tensor_util.constant_value(self.size())
                if size is not None and self.element_shape[0] is not None:
                    dim0 = size * self.element_shape[0]
            value.set_shape([dim0] + self.element_shape.dims[1:])
        return value

    @tf_should_use.should_use_result
    def unstack(self, value, name=None):
        """See TensorArray."""
        with ops.name_scope(name, 'TensorArrayUnstack', [self._handle, value]):
            num_elements = array_ops.shape(value)[0]
            return self.scatter(indices=math_ops.range(0, num_elements), value=value, name=name)

    @tf_should_use.should_use_result
    def scatter(self, indices, value, name=None):
        """See TensorArray."""
        with ops.name_scope(name, 'TensorArrayScatter', [self._handle, value, indices]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            if not context.executing_eagerly():
                self._check_element_shape(value.shape[1:])
            with self._maybe_colocate_with(value):
                flow_out = gen_data_flow_ops.tensor_array_scatter_v3(handle=self._handle, indices=indices, value=value, flow_in=self._flow, name=name)
            return build_ta_with_new_flow(self, flow_out)

    @tf_should_use.should_use_result
    def split(self, value, lengths, name=None):
        """See TensorArray."""
        with ops.name_scope(name, 'TensorArraySplit', [self._handle, value, lengths]):
            value = ops.convert_to_tensor(value, dtype=self._dtype, name='value')
            with self._maybe_colocate_with(value):
                lengths_64 = math_ops.cast(lengths, dtypes.int64)
                if not context.executing_eagerly():
                    clengths = tensor_util.constant_value(lengths_64)
                    if value.shape.dims is not None and clengths is not None:
                        if clengths.shape and clengths.max() == clengths.min():
                            self._check_element_shape(tensor_shape.TensorShape([clengths[0]]).concatenate(value.shape[1:]))
                flow_out = gen_data_flow_ops.tensor_array_split_v3(handle=self._handle, value=value, lengths=lengths_64, flow_in=self._flow, name=name)
            return build_ta_with_new_flow(self, flow_out)

    def size(self, name=None):
        """See TensorArray."""
        if not self._dynamic_size and self._size is not None:
            return ops.convert_to_tensor(self._size, dtype=dtypes.int32)
        else:
            return gen_data_flow_ops.tensor_array_size_v3(handle=self._handle, flow_in=self.flow, name=name)

    @tf_should_use.should_use_result
    def close(self, name=None):
        """See TensorArray."""
        return gen_data_flow_ops.tensor_array_close_v3(handle=self._handle, name=name)