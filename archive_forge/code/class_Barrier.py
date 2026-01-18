import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class Barrier:
    """Represents a key-value map that persists across graph executions."""

    def __init__(self, types, shapes=None, shared_name=None, name='barrier'):
        """Creates a barrier that persists across different graph executions.

    A barrier represents a key-value map, where each key is a string, and
    each value is a tuple of tensors.

    At runtime, the barrier contains 'complete' and 'incomplete'
    elements. A complete element has defined tensors for all
    components of its value tuple, and may be accessed using
    take_many. An incomplete element has some undefined components in
    its value tuple, and may be updated using insert_many.

    The barrier call `take_many` outputs values in a particular order.
    First, it only outputs completed values.  Second, the order in which
    completed values are returned matches the order in which their very
    first component was inserted into the barrier.  So, for example, for this
    sequence of insertions and removals:

      barrier = Barrier((tf.string, tf.int32), shapes=((), ()))
      barrier.insert_many(0, keys=["k1", "k2"], values=["a", "b"]).run()
      barrier.insert_many(1, keys=["k1"], values=[1]).run()
      barrier.insert_many(0, keys=["k3"], values=["c"]).run()
      barrier.insert_many(1, keys=["k3"], values=[3]).run()
      barrier.insert_many(1, keys=["k2"], values=[2]).run()

      (indices, keys, values) = barrier.take_many(2)
      (indices_val, keys_val, values0_val, values1_val) =
         session.run([indices, keys, values[0], values[1]])

    The output will be (up to permutation of "k1" and "k2"):

      indices_val == (-2**63, -2**63)
      keys_val == ("k1", "k2")
      values0_val == ("a", "b")
      values1_val == (1, 2)

    Note the key "k2" was inserted into the barrier before "k3".  Even though
    "k3" was completed first, both are complete by the time
    take_many is called.  As a result, "k2" is prioritized and "k1" and "k2"
    are returned first.  "k3" remains in the barrier until the next execution
    of `take_many`.  Since "k1" and "k2" had their first insertions into
    the barrier together, their indices are the same (-2**63).  The index
    of "k3" will be -2**63 + 1, because it was the next new inserted key.

    Args:
      types: A single dtype or a tuple of dtypes, corresponding to the
        dtypes of the tensor elements that comprise a value in this barrier.
      shapes: Optional. Constraints on the shapes of tensors in the values:
        a single tensor shape tuple; a tuple of tensor shape tuples
        for each barrier-element tuple component; or None if the shape should
        not be constrained.
      shared_name: Optional. If non-empty, this barrier will be shared under
        the given name across multiple sessions.
      name: Optional name for the barrier op.

    Raises:
      ValueError: If one of the `shapes` indicate no elements.
    """
        self._types = _as_type_list(types)
        if shapes is not None:
            shapes = _as_shape_list(shapes, self._types)
            self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
            for i, shape in enumerate(self._shapes):
                if shape.num_elements() == 0:
                    raise ValueError(f"Empty tensors are not supported, but received shape '{shape}' at index {i}")
        else:
            self._shapes = [tensor_shape.unknown_shape() for _ in self._types]
        self._barrier_ref = gen_data_flow_ops.barrier(component_types=self._types, shapes=self._shapes, shared_name=shared_name, name=name)
        if context.executing_eagerly():
            self._name = context.context().scope_name
        else:
            self._name = self._barrier_ref.op.name.split('/')[-1]

    @property
    def barrier_ref(self):
        """Get the underlying barrier reference."""
        return self._barrier_ref

    @property
    def name(self):
        """The name of the underlying barrier."""
        if context.executing_eagerly():
            return self._name
        return self._barrier_ref.op.name

    def insert_many(self, component_index, keys, values, name=None):
        """For each key, assigns the respective value to the specified component.

    This operation updates each element at component_index.

    Args:
      component_index: The component of the value that is being assigned.
      keys: A vector of keys, with length n.
      values: An any-dimensional tensor of values, which are associated with the
        respective keys. The first dimension must have length n.
      name: Optional name for the op.

    Returns:
      The operation that performs the insertion.
    Raises:
      InvalidArgumentsError: If inserting keys and values without elements.
    """
        if name is None:
            name = '%s_BarrierInsertMany' % self._name
        return gen_data_flow_ops.barrier_insert_many(self._barrier_ref, keys, values, component_index, name=name)

    def take_many(self, num_elements, allow_small_batch=False, timeout=None, name=None):
        """Takes the given number of completed elements from this barrier.

    This operation concatenates completed-element component tensors along
    the 0th dimension to make a single component tensor.

    If barrier has no completed elements, this operation will block
    until there are 'num_elements' elements to take.

    TODO(b/25743580): the semantics of `allow_small_batch` are experimental
    and may be extended to other cases in the future.

    TODO(ebrevdo): If a take_many(allow_small_batch=True) is blocking
    already when the barrier is closed, it will block for ever. Fix this
    by using asynchronous operations.

    Args:
      num_elements: The number of elements to take.
      allow_small_batch: If the barrier is closed, don't block if there are less
        completed elements than requested, but instead return all available
        completed elements.
      timeout: This specifies the number of milliseconds to block
        before returning with DEADLINE_EXCEEDED. (This option is not
        supported yet.)
      name: A name for the operation (optional).

    Returns:
      A tuple of (index, key, value_list).
      "index" is a int64 tensor of length num_elements containing the
        index of the insert_many call for which the very first component of
        the given element was inserted into the Barrier, starting with
        the value -2**63.  Note, this value is different from the
        index of the insert_many call for which the element was completed.
      "key" is a string tensor of length num_elements containing the keys.
      "value_list" is a tuple of tensors, each one with size num_elements
        in the 0th dimension for each component in the barrier's values.

    """
        if name is None:
            name = '%s_BarrierTakeMany' % self._name
        ret = gen_data_flow_ops.barrier_take_many(self._barrier_ref, num_elements, self._types, allow_small_batch, timeout, name=name)
        if not context.executing_eagerly():
            op = ret[0].op
            if allow_small_batch:
                batch_dim = None
            else:
                batch_dim = tensor_shape.Dimension(tensor_util.constant_value(op.inputs[1]))
            op.outputs[0].set_shape(tensor_shape.TensorShape([batch_dim]))
            op.outputs[1].set_shape(tensor_shape.TensorShape([batch_dim]))
            for output, shape in zip(op.outputs[2:], self._shapes):
                output.set_shape(tensor_shape.TensorShape([batch_dim]).concatenate(shape))
        return ret

    def close(self, cancel_pending_enqueues=False, name=None):
        """Closes this barrier.

    This operation signals that no more new key values will be inserted in the
    given barrier. Subsequent InsertMany operations with new keys will fail.
    InsertMany operations that just complement already existing keys with other
    components, will continue to succeed. Subsequent TakeMany operations will
    continue to succeed if sufficient elements remain in the barrier. Subsequent
    TakeMany operations that would block will fail immediately.

    If `cancel_pending_enqueues` is `True`, all pending requests to the
    underlying queue will also be canceled, and completing of already
    started values is also not acceptable anymore.

    Args:
      cancel_pending_enqueues: (Optional.) A boolean, defaulting to
        `False` (described above).
      name: Optional name for the op.

    Returns:
      The operation that closes the barrier.
    """
        if name is None:
            name = '%s_BarrierClose' % self._name
        return gen_data_flow_ops.barrier_close(self._barrier_ref, cancel_pending_enqueues=cancel_pending_enqueues, name=name)

    def ready_size(self, name=None):
        """Compute the number of complete elements in the given barrier.

    Args:
      name: A name for the operation (optional).

    Returns:
      A single-element tensor containing the number of complete elements in the
      given barrier.
    """
        if name is None:
            name = '%s_BarrierReadySize' % self._name
        return gen_data_flow_ops.barrier_ready_size(self._barrier_ref, name=name)

    def incomplete_size(self, name=None):
        """Compute the number of incomplete elements in the given barrier.

    Args:
      name: A name for the operation (optional).

    Returns:
      A single-element tensor containing the number of incomplete elements in
      the given barrier.
    """
        if name is None:
            name = '%s_BarrierIncompleteSize' % self._name
        return gen_data_flow_ops.barrier_incomplete_size(self._barrier_ref, name=name)