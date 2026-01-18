import copy
import math
from typing import Sequence
import weakref
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
class ShardedVariableMixin(trackable.Trackable):
    """Mixin for ShardedVariable."""

    def __init__(self, variables, name='ShardedVariable'):
        """Treats `variables` as shards of a larger Variable.

    Example:

    ```
    variables = [
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
      tf.Variable(..., shape=(15, 100), dtype=tf.float32),
      tf.Variable(..., shape=(5, 100), dtype=tf.float32)
    ]
    sharded_variable = ShardedVariableMixin(variables)
    assert sharded_variable.shape.as_list() == [30, 100]
    ```

    Args:
      variables: A list of `ResourceVariable`s that comprise this sharded
        variable. Variables should not be shared between different
        `ShardedVariableMixin` objects.
      name: String. Name of this container. Defaults to "ShardedVariable".
    """
        super(ShardedVariableMixin, self).__init__()
        self._variables = variables
        self._name = name
        if not isinstance(variables, Sequence) or not variables or any((not isinstance(v, variables_lib.Variable) for v in variables)):
            raise TypeError(f'Argument `variables` should be a non-empty list of `variables.Variable`s. Received {variables}')
        var_dtypes = {v.dtype for v in variables}
        if len(var_dtypes) > 1:
            raise ValueError(f'All elements in argument `variables` must have the same dtype. Received dtypes: {[v.dtype for v in variables]}')
        first_var = variables[0]
        self._dtype = first_var.dtype
        higher_dim_shapes = {tuple(v.shape.as_list()[1:]) for v in variables}
        if len(higher_dim_shapes) > 1:
            raise ValueError(f'All elements in argument `variables` must have the same shapes except for the first axis. Received shapes: {[v.shape for v in variables]}')
        first_dim = sum((int(v.shape.as_list()[0]) for v in variables))
        self._shape = tensor_shape.TensorShape([first_dim] + first_var.shape.as_list()[1:])
        for v in variables:
            v._sharded_container = weakref.ref(self)
        self._var_offsets = [[0 for _ in range(len(first_var.shape))] for _ in range(len(variables))]
        for i in range(1, len(variables)):
            self._var_offsets[i][0] += self._var_offsets[i - 1][0] + variables[i - 1].shape.as_list()[0]
        save_slice_info = [v._get_save_slice_info() for v in variables]
        if any((slice_info is not None for slice_info in save_slice_info)):
            raise ValueError(f'`SaveSliceInfo` should not be set for all elements in argument `variables`. `ShardedVariable` will infer `SaveSliceInfo` according to the order of the elements `variables`. Received save slice info {save_slice_info}')
        self._saving_variable = resource_variable_ops.UninitializedVariable(shape=self._shape, dtype=self._dtype, name=self._name, trainable=self._variables[0].trainable, synchronization=variables_lib.VariableSynchronization.NONE, aggregation=variables_lib.VariableAggregation.NONE)

    def __iter__(self):
        """Return an iterable for accessing the underlying sharded variables."""
        return iter(self._variables)

    def __getitem__(self, slice_spec):
        """Extracts the specified region as a Tensor from the sharded variable.

    The API contract is identical to `Tensor.__getitem__`. Assignment to the
    sliced range is not yet supported.

    Args:
      slice_spec: The arguments to __getitem__, specifying the global slicing of
        the sharded variable.

    Returns:
      The appropriate slice of tensor based on `slice_spec`.

    Raises:
      IndexError: If a slice index is out of bound.
      TypeError: If `spec_spec` contains Tensor.
    """
        if isinstance(slice_spec, bool) or (isinstance(slice_spec, tensor_lib.Tensor) and slice_spec.dtype == dtypes.bool) or (isinstance(slice_spec, np.ndarray) and slice_spec.dtype == bool):
            tensor = _var_to_tensor(self)
            return array_ops.boolean_mask(tensor=tensor, mask=slice_spec)
        if not isinstance(slice_spec, (list, tuple)):
            slice_spec = (slice_spec,)
        s = slice_spec[0]
        if isinstance(s, slice):
            first_dim_slice_specs = self._decompose_slice_spec(s)
            values = []
            for i, var in enumerate(self._variables):
                if first_dim_slice_specs[i] is not None:
                    all_dim_slice_spec = (first_dim_slice_specs[i],) + slice_spec[1:]
                    values.append(var[all_dim_slice_spec])
            if s.step is not None and s.step < 0:
                values.reverse()
            if not values:
                return constant_op.constant([], dtype=self._dtype, shape=(0,) + self._shape[1:])
            return array_ops.concat(values, axis=0)
        elif s is Ellipsis:
            return array_ops.concat([var[slice_spec] for var in self._variables], axis=0)
        elif s is array_ops.newaxis:
            return array_ops.concat([var[slice_spec[1:]] for var in self._variables], axis=0)[array_ops.newaxis]
        else:
            if isinstance(s, tensor_lib.Tensor):
                raise TypeError('ShardedVariable: using Tensor for indexing is not allowed.')
            if s < 0:
                s += self._shape[0]
            if s < 0 or s >= self._shape[0]:
                raise IndexError(f'ShardedVariable: slice index {s} of dimension 0 out of bounds.')
            for i in range(len(self._variables)):
                if i == len(self._variables) - 1 or (s > self._var_offsets[i][0] and s < self._var_offsets[i + 1][0]):
                    return self._variables[i][(s - self._var_offsets[i][0],) + slice_spec[1:]]

    def _decompose_slice_spec(self, slice_spec):
        """Decompose a global slice_spec into a list of per-variable slice_spec.

    `ShardedVariable` only supports first dimension partitioning, thus
    `slice_spec` must be for first dimension.

    Args:
      slice_spec: A python `slice` object that specifies the global slicing.

    Returns:
      A list of python `slice` objects or None specifying the local slicing for
      each component variable. None means no slicing.

    For example, given component variables:
      v0 = [0, 1, 2]
      v1 = [3, 4, 5]
      v2 = [6, 7, 8, 9]

    If `slice_spec` is slice(start=None, stop=None, step=None), we will have:
      v0[returned[0]] = [0, 1, 2]
      v1[returned[1]] = [3, 4, 5]
      v2[returned[2]] = [6, 7, 8, 9]
    If `slice_spec` is slice(start=2, stop=8, step=3), we will have:
      v0[returned[0]] = [2]
      v1[returned[1]] = [5]
      returned[2] == None
    If `slice_spec` is slice(start=9, stop=3, step=-2), we will have:
      returned[0] == None
      v1[returned[1]] = [5]
      v2[returned[2]] = [9, 7]
    """
        if isinstance(slice_spec.start, tensor_lib.Tensor) or isinstance(slice_spec.stop, tensor_lib.Tensor) or isinstance(slice_spec.step, tensor_lib.Tensor):
            raise TypeError('ShardedVariable: using Tensor in slice_spec is not allowed. Please file a feature request with the TensorFlow team.')
        result = []
        slice_step = slice_spec.step if slice_spec.step is not None else 1
        if slice_step == 0:
            raise ValueError('slice step cannot be zero')
        slice_start = slice_spec.start
        if slice_start is None:
            slice_start = 0 if slice_step > 0 else self._shape[0] - 1
        elif slice_start < 0:
            slice_start += self._shape[0]
        slice_end = slice_spec.stop
        if slice_end is None:
            slice_end = self._shape[0] if slice_step > 0 else -1
        elif slice_end < 0:
            slice_end += self._shape[0]
        cur = slice_start
        if slice_step > 0:
            for i in range(len(self._var_offsets)):
                var_start = self._var_offsets[i][0]
                var_end = self._var_offsets[i + 1][0] if i < len(self._var_offsets) - 1 else self._shape[0]
                if cur < var_start:
                    cur += slice_step * int(math.ceil((var_start - cur) / slice_step))
                if cur >= var_end or cur >= slice_end:
                    result.append(None)
                else:
                    start = cur - var_start
                    end = min(slice_end, var_end) - var_start
                    result.append(slice(start, end, slice_step))
        else:
            for i in range(len(self._var_offsets) - 1, -1, -1):
                var_start = self._var_offsets[i][0]
                var_end = self._var_offsets[i + 1][0] if i < len(self._var_offsets) - 1 else self._shape[0]
                if cur >= var_end:
                    cur += slice_step * int(math.ceil((var_end - cur - 1) / slice_step))
                if cur < var_start or cur <= slice_end:
                    result.append(None)
                else:
                    start = cur - var_start
                    if slice_end >= var_start:
                        end = slice_end - var_start
                    else:
                        end = None
                    result.append(slice(start, end, slice_step))
            result.reverse()
        return result

    @property
    def _type_spec(self):
        return ShardedVariableSpec(*(resource_variable_ops.VariableSpec(v.shape, v.dtype) for v in self._variables))

    @property
    def variables(self):
        """The list of `Variable`s that make up the shards of this object."""
        if save_context.in_save_context():
            return [self._saving_variable]
        return self._variables

    @property
    def name(self):
        """The name of this object. Used for checkpointing."""
        return self._name

    @property
    def dtype(self):
        """The dtype of all `Variable`s in this object."""
        return self._dtype

    @property
    def shape(self):
        """The overall shape, combining all shards along axis `0`."""
        return self._shape

    def assign(self, value, use_locking=None, name=None, read_value=True):
        for i, v in enumerate(self._variables):
            v.assign(array_ops.slice(value, self._var_offsets[i], v.shape.as_list()))
        return self

    def assign_add(self, delta, use_locking=False, name=None, read_value=True):
        for i, v in enumerate(self._variables):
            v.assign_add(array_ops.slice(delta, self._var_offsets[i], v.shape.as_list()))
        return self

    def assign_sub(self, delta, use_locking=False, name=None, read_value=True):
        for i, v in enumerate(self._variables):
            v.assign_sub(array_ops.slice(delta, self._var_offsets[i], v.shape.as_list()))
        return self

    def _decompose_indices(self, indices):
        """Decompose a global 1D indices into a list of per-variable indices."""
        if indices.shape.rank != 1:
            raise ValueError(f'ShardedVariable: indices must be 1D Tensor for sparse operations. Received shape: {indices.shape}')
        base = self._shape[0] // len(self._variables)
        extra = self._shape[0] % len(self._variables)
        expect_first_dim = [base] * len(self._variables)
        for i in range(extra):
            expect_first_dim[i] = expect_first_dim[i] + 1
        actual_first_dim = [v.shape.as_list()[0] for v in self._variables]
        if expect_first_dim != actual_first_dim:
            raise NotImplementedError('scater_xxx ops are not supported in ShardedVariale that does not conform to "div" sharding')
        partition_assignments = math_ops.maximum(indices // (base + 1), (indices - extra) // base)
        local_indices = array_ops.where(partition_assignments < extra, indices % (base + 1), (indices - extra) % base)
        partition_assignments = math_ops.cast(partition_assignments, dtypes.int32)
        per_var_indices = data_flow_ops.dynamic_partition(local_indices, partition_assignments, len(self._variables))
        return (per_var_indices, partition_assignments)

    def _decompose_indexed_slices(self, indexed_slices):
        """Decompose a global `IndexedSlices` into a list of per-variable ones."""
        per_var_indices, partition_assignments = self._decompose_indices(indexed_slices.indices)
        per_var_values = data_flow_ops.dynamic_partition(indexed_slices.values, partition_assignments, len(self._variables))
        return [indexed_slices_lib.IndexedSlices(values=per_var_values[i], indices=per_var_indices[i]) for i in range(len(self._variables))]

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        """Implements tf.Variable.scatter_add."""
        per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
        for i, v in enumerate(self._variables):
            new_name = None
            if name is not None:
                new_name = '{}/part_{}'.format(name, i)
            v.scatter_add(per_var_sparse_delta[i], name=new_name)
        return self

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        """Implements tf.Variable.scatter_div."""
        per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
        for i, v in enumerate(self._variables):
            new_name = None
            if name is not None:
                new_name = '{}/part_{}'.format(name, i)
            v.scatter_div(per_var_sparse_delta[i], name=new_name)
        return self

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        """Implements tf.Variable.scatter_max."""
        per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
        for i, v in enumerate(self._variables):
            new_name = None
            if name is not None:
                new_name = '{}/part_{}'.format(name, i)
            v.scatter_max(per_var_sparse_delta[i], name=new_name)
        return self

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        """Implements tf.Variable.scatter_min."""
        per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
        for i, v in enumerate(self._variables):
            new_name = None
            if name is not None:
                new_name = '{}/part_{}'.format(name, i)
            v.scatter_min(per_var_sparse_delta[i], name=new_name)
        return self

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        """Implements tf.Variable.scatter_mul."""
        per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
        for i, v in enumerate(self._variables):
            new_name = None
            if name is not None:
                new_name = '{}/part_{}'.format(name, i)
            v.scatter_mul(per_var_sparse_delta[i], name=new_name)
        return self

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        """Implements tf.Variable.scatter_sub."""
        per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
        for i, v in enumerate(self._variables):
            new_name = None
            if name is not None:
                new_name = '{}/part_{}'.format(name, i)
            v.scatter_sub(per_var_sparse_delta[i], name=new_name)
        return self

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        """Implements tf.Variable.scatter_update."""
        per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
        for i, v in enumerate(self._variables):
            new_name = None
            if name is not None:
                new_name = '{}/part_{}'.format(name, i)
            v.scatter_update(per_var_sparse_delta[i], name=new_name)
        return self

    def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
        """Implements tf.Variable.batch_scatter_update."""
        per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
        for i, v in enumerate(self._variables):
            new_name = None
            if name is not None:
                new_name = '{}/part_{}'.format(name, i)
            v.batch_scatter_update(per_var_sparse_delta[i], name=new_name)
        return self

    def sparse_read(self, indices, name=None):
        """Implements tf.Variable.sparse_read."""
        per_var_indices, _ = self._decompose_indices(indices)
        result = []
        for i, v in enumerate(self._variables):
            new_name = None
            if name is not None:
                new_name = '{}/part_{}'.format(name, i)
            result.append(v.sparse_read(per_var_indices[i], name=new_name))
        return array_ops.concat(result, axis=0)

    def _gather_saveables_for_checkpoint(self):
        """Return a `Saveable` for each shard. See `Trackable`."""

        def _saveable_factory(name=self.name):
            """Creates `SaveableObject`s for this `ShardedVariable`."""
            saveables = []
            dims = len(self._variables[0].shape)
            var_offset = [0 for _ in range(dims)]
            for v in self._variables:
                save_slice_info = variables_lib.Variable.SaveSliceInfo(full_name=self.name, full_shape=self.shape.as_list(), var_offset=copy.copy(var_offset), var_shape=v.shape.as_list())
                saveables.append(saveable_object_util.ResourceVariableSaveable(v, save_slice_info.spec, name))
                var_offset[0] += int(v.shape[0])
            return saveables
        return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

    def _export_to_saved_model_graph(self, object_map, tensor_map, options, **kwargs):
        """For implementing `Trackable`."""
        resource_list = []
        for v in self._variables + [self._saving_variable]:
            resource_list.extend(v._export_to_saved_model_graph(object_map, tensor_map, options, **kwargs))
        object_map[self] = ShardedVariable([object_map[self._saving_variable]], name=self.name)
        return resource_list

    @property
    def _unique_id(self):
        return self.variables[0]._unique_id.replace('part_0', 'sharded')

    @property
    def _distribute_strategy(self):
        return self.variables[0]._distribute_strategy

    @property
    def _shared_name(self):
        return self._name

    @property
    def is_sharded_variable(self):
        return True

    def numpy(self):
        """Copies the values in this ShardedVariable to a NumPy array.

    First converts to a single Tensor using the registered conversion function,
    which concatenates the shards, then uses Tensor.numpy() to convert to
    a NumPy array.

    Returns:
      A NumPy array of the same shape and dtype.
    """
        return _var_to_tensor(self).numpy()