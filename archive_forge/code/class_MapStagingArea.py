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
class MapStagingArea(BaseStagingArea):
    """A `MapStagingArea` is a TensorFlow data structure that stores tensors
  across multiple steps, and exposes operations that can put and get tensors.

  Each `MapStagingArea` element is a (key, value) pair.
  Only int64 keys are supported, other types should be
  hashed to produce a key.
  Values are a tuple of one or more tensors.
  Each tuple component has a static dtype,
  and may have a static shape.

  The capacity of a `MapStagingArea` may be bounded or unbounded.
  It supports multiple concurrent producers and consumers; and
  provides exactly-once delivery.

  Each value tuple of a `MapStagingArea` is a fixed-length tuple of tensors
  whose
  dtypes are described by `dtypes`, and whose shapes are optionally described
  by the `shapes` argument.

  If the `shapes` argument is specified, each component of a staging area
  element must have the respective fixed shape. If it is
  unspecified, different elements may have different shapes,

  It behaves like an associative container with support for:

   - put(key, values)
   - peek(key)         like dict.get(key)
   - get(key)          like dict.pop(key)
   - get(key=None)     like dict.popitem()
   - size()
   - clear()

  If ordered a tree structure ordered by key will be used and
  get(key=None) will remove (key, value) pairs in increasing key order.
  Otherwise a hashtable

  It can be configured with a capacity in which case
  put(key, values) will block until space becomes available.

  Similarly, it can be configured with a memory limit which
  will block put(key, values) until space is available.
  This is mostly useful for limiting the number of tensors on
  devices such as GPUs.

  All get() and peek() commands block if the requested
  (key, value) pair is not present in the staging area.

  Partial puts are supported and will be placed in an incomplete
  map until such time as all values associated with the key have
  been inserted. Once completed, this (key, value) pair will be
  inserted into the map. Data in the incomplete map
  counts towards the memory limit, but not towards capacity limit.

  Partial gets from the map are also supported.
  This removes the partially requested tensors from the entry,
  but the entry is only removed from the map once all tensors
  associated with it are removed.
  """

    def __init__(self, dtypes, shapes=None, names=None, shared_name=None, ordered=False, capacity=0, memory_limit=0):
        """Args:

      dtypes:  A list of types.  The length of dtypes must equal the number
        of tensors in each element.
      capacity: (Optional.) Maximum number of elements.
        An integer. If zero, the Staging Area is unbounded
      memory_limit: (Optional.) Maximum number of bytes of all tensors
        in the Staging Area (excluding keys).
        An integer. If zero, the Staging Area is unbounded
      ordered: (Optional.) If True the underlying data structure
        is a tree ordered on key. Otherwise assume a hashtable.
      shapes: (Optional.) Constraints on the shapes of tensors in an element.
        A list of shape tuples or None. This list is the same length
        as dtypes.  If the shape of any tensors in the element are constrained,
        all must be; shapes can be None if the shapes should not be constrained.
      names: (Optional.) If provided, the `get()` and
        `put()` methods will use dictionaries with these names as keys.
        Must be None or a list or tuple of the same length as `dtypes`.
      shared_name: (Optional.) A name to be used for the shared object. By
        passing the same name to two different python objects they will share
        the underlying staging area. Must be a string.

    Raises:
      ValueError: If one of the arguments is invalid.

    """
        super(MapStagingArea, self).__init__(dtypes, shapes, names, shared_name, capacity, memory_limit)
        self._ordered = ordered
        if ordered:
            self._put_fn = gen_data_flow_ops.ordered_map_stage
            self._pop_fn = gen_data_flow_ops.ordered_map_unstage
            self._popitem_fn = gen_data_flow_ops.ordered_map_unstage_no_key
            self._peek_fn = gen_data_flow_ops.ordered_map_peek
            self._size_fn = gen_data_flow_ops.ordered_map_size
            self._incomplete_size_fn = gen_data_flow_ops.ordered_map_incomplete_size
            self._clear_fn = gen_data_flow_ops.ordered_map_clear
        else:
            self._put_fn = gen_data_flow_ops.map_stage
            self._pop_fn = gen_data_flow_ops.map_unstage
            self._popitem_fn = gen_data_flow_ops.map_unstage_no_key
            self._peek_fn = gen_data_flow_ops.map_peek
            self._size_fn = gen_data_flow_ops.map_size
            self._incomplete_size_fn = gen_data_flow_ops.map_incomplete_size
            self._clear_fn = gen_data_flow_ops.map_clear

    def put(self, key, vals, indices=None, name=None):
        """Create an op that stores the (key, vals) pair in the staging area.

    Incomplete puts are possible, preferably using a dictionary for vals
    as the appropriate dtypes and shapes can be inferred from the value names
    dictionary key values. If vals is a list or tuple, indices must
    also be specified so that the op knows at which element position
    to perform the insert.

    This operation will block if the capacity or memory limit of this
    container is reached.

    Args:
        key: Key associated with the data
        vals: Tensor (or a dict/tuple of Tensors) to place
                into the staging area.
        indices: (Optional) if vals is a tuple/list, this is required.
        name: A name for the operation (optional)

    Returns:
        The created op

    Raises:
        ValueError: If the number or type of inputs don't match the staging
        area.
    """
        with ops.name_scope(name, '%s_put' % self._name, self._scope_vals(vals)) as scope:
            vals, indices = self._check_put_dtypes(vals, indices)
            with ops.colocate_with(self._coloc_op):
                op = self._put_fn(key, indices, vals, dtypes=self._dtypes, shared_name=self._name, name=scope, capacity=self._capacity, memory_limit=self._memory_limit)
        return op

    def _get_indices_and_dtypes(self, indices=None):
        if indices is None:
            indices = list(range(len(self._dtypes)))
        if not isinstance(indices, (tuple, list)):
            raise TypeError(f'Invalid indices type {type(indices)}')
        if len(indices) == 0:
            raise ValueError('Empty indices')
        if all((isinstance(i, str) for i in indices)):
            if self._names is None:
                raise ValueError(f'String indices provided {indices}, but this Staging Area was not created with names.')
            try:
                indices = [self._names.index(n) for n in indices]
            except ValueError:
                raise ValueError(f'Named index not in Staging Area names {self._names}')
        elif all((isinstance(i, int) for i in indices)):
            pass
        else:
            raise TypeError(f'Mixed types in indices {indices}. May only be str or int')
        dtypes = [self._dtypes[i] for i in indices]
        return (indices, dtypes)

    def peek(self, key, indices=None, name=None):
        """Peeks at staging area data associated with the key.

    If the key is not in the staging area, it will block
    until the associated (key, value) is inserted.

    Args:
        key: Key associated with the required data
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)

    Returns:
        The created op
    """
        if name is None:
            name = '%s_pop' % self._name
        indices, dtypes = self._get_indices_and_dtypes(indices)
        with ops.colocate_with(self._coloc_op):
            result = self._peek_fn(key, shared_name=self._name, indices=indices, dtypes=dtypes, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
        return self._get_return_value(result, indices)

    def get(self, key=None, indices=None, name=None):
        """If the key is provided, the associated (key, value) is returned from the staging area.

    If the key is not in the staging area, this method will block until
    the associated (key, value) is inserted.
    If no key is provided and the staging area is ordered,
    the (key, value) with the smallest key will be returned.
    Otherwise, a random (key, value) will be returned.

    If the staging area is empty when this operation executes,
    it will block until there is an element to dequeue.

    Args:
        key: Key associated with the required data (Optional)
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)

    Returns:
        The created op
    """
        if key is None:
            return self._popitem(indices=indices, name=name)
        else:
            return self._pop(key, indices=indices, name=name)

    def _pop(self, key, indices=None, name=None):
        """Remove and return the associated (key, value) is returned from the staging area.

    If the key is not in the staging area, this method will block until
    the associated (key, value) is inserted.
    Args:
        key: Key associated with the required data
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)

    Returns:
        The created op
    """
        if name is None:
            name = '%s_get' % self._name
        indices, dtypes = self._get_indices_and_dtypes(indices)
        with ops.colocate_with(self._coloc_op):
            result = self._pop_fn(key, shared_name=self._name, indices=indices, dtypes=dtypes, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
        return (key, self._get_return_value(result, indices))

    def _popitem(self, indices=None, name=None):
        """If the staging area is ordered, the (key, value) with the smallest key will be returned.

    Otherwise, a random (key, value) will be returned.
    If the staging area is empty when this operation executes,
    it will block until there is an element to dequeue.

    Args:
        key: Key associated with the required data
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)

    Returns:
        The created op
    """
        if name is None:
            name = '%s_get_nokey' % self._name
        indices, dtypes = self._get_indices_and_dtypes(indices)
        with ops.colocate_with(self._coloc_op):
            key, result = self._popitem_fn(shared_name=self._name, indices=indices, dtypes=dtypes, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
        key = self._create_device_transfers(key)[0]
        result = self._get_return_value(result, indices)
        return (key, result)

    def size(self, name=None):
        """Returns the number of elements in the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
        if name is None:
            name = '%s_size' % self._name
        return self._size_fn(shared_name=self._name, name=name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)

    def incomplete_size(self, name=None):
        """Returns the number of incomplete elements in the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
        if name is None:
            name = '%s_incomplete_size' % self._name
        return self._incomplete_size_fn(shared_name=self._name, name=name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)

    def clear(self, name=None):
        """Clears the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
        if name is None:
            name = '%s_clear' % self._name
        return self._clear_fn(shared_name=self._name, name=name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)