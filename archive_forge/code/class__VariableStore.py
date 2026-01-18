import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _VariableStore:
    """Variable store that carries a number of named Variables.

  New variable names and new variables can be created; all stored
  variables are initialized with the initializer passed to __init__.

  Attributes:
    vars: a dictionary with string names (same as passed in GetVar) as keys and
      the corresponding TensorFlow Variables as values.
  """
    __slots__ = ['_vars', '_partitioned_vars', '_store_eager_variables']

    def __init__(self):
        """Create a variable store."""
        self._vars = {}
        self._partitioned_vars = {}
        self._store_eager_variables = False

    def get_variable(self, name, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, reuse=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, custom_getter=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
        """Gets an existing variable with these parameters or create a new one.

    If a variable with the given name is already stored, we return the stored
    variable. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want
    variables to be created if they don't exist or returned if they do.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If a partitioner is provided, a `PartitionedVariable` is returned.
    Accessing this object as a `Tensor` returns the shards concatenated along
    the partition axis.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
      initializer: Initializer for the variable.
      regularizer: A (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of
        variables. When eager execution is enabled  this argument is always
        forced to be False.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`). `trainable`
        defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys to add the `Variable` to.
        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the `Variable` reside, to
        deduplicate copying through `Switch` and other conditional statements.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and dtype of the `Variable` to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
      use_resource: If False, creates a regular Variable. If True, creates
        instead an experimental ResourceVariable which has well-defined
        semantics. Defaults to False (will later change to True). When eager
        execution is enabled this argument is always forced to be true.
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. The signature
        of `custom_getter` should match that of this method,
        but the most future-proof version will allow for changes: `def
          custom_getter(getter, *args, **kwargs)`.  Direct access to
        all `get_variable` parameters is also allowed: `def
          custom_getter(getter, name, *args, **kwargs)`.  A simple identity
        custom getter that simply creates variables with modified names is:
          ```python
        def custom_getter(getter, name, *args, **kwargs): return getter(name +
          '_suffix', *args, **kwargs) ```
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

    Returns:
      The created or existing `Variable` (or `PartitionedVariable`, if a
      partitioner was used).

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        or when violating reuse during variable creation.
      RuntimeError: when eager execution is enabled and not called from an
        EagerVariableStore.
    """
        if custom_getter is not None and (not callable(custom_getter)):
            raise ValueError('Passed a custom_getter which is not callable: %s' % custom_getter)
        with ops.init_scope():
            if context.executing_eagerly():
                use_resource = True
        if context.executing_eagerly():
            if not self._store_eager_variables and reuse:
                raise RuntimeError('When eager execution is enabled variable reuse is only supported when an EagerVariableStore is active. See the documentation on EagerVariableStore for example usage.')
            if self._store_eager_variables:
                reuse = AUTO_REUSE
        try:
            dtype = dtype.base_dtype
        except AttributeError:
            pass

        def _true_getter(name, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, reuse=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
            is_scalar = shape is not None and isinstance(shape, collections_abc.Sequence) and (not shape)
            if partitioner is not None and (not is_scalar):
                if not callable(partitioner):
                    raise ValueError('Partitioner must be callable, but received: %s' % partitioner)
                with ops.name_scope(None):
                    return self._get_partitioned_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
            if reuse is True and partitioner is None and (name in self._partitioned_vars):
                return self._get_partitioned_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=None, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
            if '%s/part_0' % name in self._vars:
                raise ValueError('No partitioner was provided, but a partitioned version of the variable was found: %s/part_0. Perhaps a variable of the same name was already created with partitioning?' % name)
            return self._get_single_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
        synchronization, aggregation, trainable = variables.validate_synchronization_aggregation_trainable(synchronization, aggregation, trainable, name)
        if custom_getter is not None:
            custom_getter_kwargs = {'getter': _true_getter, 'name': name, 'shape': shape, 'dtype': dtype, 'initializer': initializer, 'regularizer': regularizer, 'reuse': reuse, 'trainable': trainable, 'collections': collections, 'caching_device': caching_device, 'partitioner': partitioner, 'validate_shape': validate_shape, 'use_resource': use_resource, 'synchronization': synchronization, 'aggregation': aggregation}
            if 'constraint' in function_utils.fn_args(custom_getter) or function_utils.has_kwargs(custom_getter):
                custom_getter_kwargs['constraint'] = constraint
            return custom_getter(**custom_getter_kwargs)
        else:
            return _true_getter(name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)

    def _get_partitioned_variable(self, name, partitioner, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, reuse=None, trainable=None, collections=None, caching_device=None, validate_shape=True, use_resource=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
        """Gets or creates a sharded variable list with these parameters.

    The `partitioner` must be a callable that accepts a fully defined
    `TensorShape` and returns a sequence of integers (the `partitions`).
    These integers describe how to partition the given sharded `Variable`
    along the given dimension.  That is, `partitions[1] = 3` means split
    the `Variable` into 3 shards along dimension 1.  Currently, sharding along
    only one axis is supported.

    If the list of variables with the given name (prefix) is already stored,
    we return the stored variables. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want
    variables to be created if they don't exist or returned if they do.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If the initializer is a callable, then it will be called for each
    shard.  Otherwise the initializer should match the shape of the entire
    sharded Variable, and it will be sliced accordingly for each shard.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: the name of the new or existing sharded variable.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and `dtype` of the Variable to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      shape: shape of the new or existing sharded variable.
      dtype: type of the new or existing sharded variable (defaults to
        `DT_FLOAT`).
      initializer: initializer for the sharded variable.
      regularizer: a (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of
        variables.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      collections: List of graph collections keys to add the Variable to.
        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
      use_resource: If False, creates a regular Variable. If True, creates an
        experimental ResourceVariable which has well-defined semantics. Defaults
        to False (will later change to True).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

    Returns:
      A `PartitionedVariable` object.

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        when violating reuse during variable creation, or if an existing
        sharded variable exists for the given name but with different sharding.
    """
        initializing_from_value = initializer is not None and isinstance(initializer, tensor.Tensor)
        if name in self._vars:
            raise ValueError('A partitioner was provided, but an unpartitioned version of the variable was found: %s.  Perhaps a variable of the same name was already created without partitioning?' % name)
        shape = tensor_shape.as_shape(shape)
        if initializing_from_value:
            shape = shape.merge_with(initializer.get_shape())
        partitions = None
        if not reuse or partitioner:
            partitions = _call_partitioner(partitioner, shape, dtype)
        if name in self._partitioned_vars:
            if reuse is False:
                raise ValueError('Partitioned variable with name %s already exists. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?' % name)
            existing_var = self._partitioned_vars[name]
            if not shape.is_compatible_with(existing_var.get_shape()):
                raise ValueError('Trying to reuse partitioned variable %s, but specified shape %s and found shape %s.' % (name, shape, existing_var.get_shape()))
            if not dtype.is_compatible_with(existing_var.dtype):
                raise ValueError('Trying to reuse partitioned variable %s, but specified dtype %s and found dtype %s.' % (name, dtype.name, existing_var.dtype.name))
            if partitions is not None and existing_var._get_partitions() != partitions:
                raise ValueError('Trying to reuse partitioned variable %s, but specified partitions %s and found partitions %s.' % (name, partitions, existing_var._get_partitions()))
            return existing_var
        if reuse is True:
            raise ValueError('PartitionedVariable %s does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=False or reuse=tf.AUTO_REUSE in VarScope?' % name)
        slice_dim, num_slices = _get_slice_dim_and_num_slices(partitions)
        if '%s/part_0' % name in self._vars:
            if '%s/part_%d' % (name, num_slices - 1) not in self._vars:
                raise ValueError('Partitioner returned a different partitioning than what was already found.  Partitioner returned %d shards, and shard %s/part_0 was found, but %s/part_%d was not.' % (num_slices, name, name, num_slices - 1))
            if '%s/part_%d' % (name, num_slices) in self._vars:
                raise ValueError('Partitioner returned a different partitioning than what was already found.  Partitioner returned %d shards, and shard %s/part_0 was found, but so was the extra shard %s/part_%d.' % (num_slices, name, name, num_slices))
        vs = []
        for i, (var_offset, var_shape) in enumerate(_iter_slices(shape.as_list(), num_slices, slice_dim)):
            partition_info = _PartitionInfo(full_shape=shape.as_list(), var_offset=var_offset)
            var_full_name = '%s/part_%d' % (name, i)
            with ops.name_scope(var_full_name + '/PartitionedInitializer', skip_on_eager=False):
                if initializer is None:
                    init, initializing_from_value = self._get_default_initializer(name=name, shape=shape, dtype=dtype)
                    if initializing_from_value:
                        init_shape = None
                    else:
                        init_shape = var_shape
                elif callable(initializer):
                    init = initializer
                    init_shape = var_shape
                elif isinstance(initializer, tensor.Tensor):
                    init = array_ops.slice(initializer, var_offset, var_shape)
                    dtype = init.dtype.base_dtype
                    init_shape = None
                else:
                    init = ops.convert_to_tensor(initializer, dtype=dtype)
                    init = array_ops.slice(init, var_offset, var_shape)
                    init_shape = None
            with ops.name_scope(None):
                var = self._get_single_variable(name=var_full_name, shape=init_shape, dtype=dtype, initializer=init, partition_info=partition_info, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
            var._set_save_slice_info(variables.Variable.SaveSliceInfo(name, shape.as_list(), var_offset, var_shape))
            vs.append(var)
        partitioned_var = variables.PartitionedVariable(name=name, shape=shape, dtype=dtype, variable_list=vs, partitions=partitions)
        if not context.executing_eagerly() or self._store_eager_variables:
            self._partitioned_vars[name] = partitioned_var
        return partitioned_var

    def _get_single_variable(self, name, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, partition_info=None, reuse=None, trainable=None, collections=None, caching_device=None, validate_shape=True, use_resource=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
        """Get or create a single Variable (e.g.

    a shard or entire variable).

    See the documentation of get_variable above (ignore partitioning components)
    for details.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.
      initializer: see get_variable.
      regularizer: see get_variable.
      partition_info: _PartitionInfo object.
      reuse: see get_variable.
      trainable: see get_variable.
      collections: see get_variable.
      caching_device: see get_variable.
      validate_shape: see get_variable.
      use_resource: see get_variable.
      constraint: see get_variable.
      synchronization: see get_variable.
      aggregation: see get_variable.

    Returns:
      A Variable.  See documentation of get_variable above.

    Raises:
      ValueError: See documentation of get_variable above.
    """
        initializing_from_value = False
        if initializer is not None and (not callable(initializer)):
            initializing_from_value = True
        if shape is not None and initializing_from_value:
            raise ValueError('If initializer is a constant, do not specify shape.')
        dtype = dtypes.as_dtype(dtype)
        shape = tensor_shape.as_shape(shape)
        if name in self._vars:
            if reuse is False:
                var = self._vars[name]
                err_msg = 'Variable %s already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?' % name
                if isinstance(var, resource_variable_ops.ResourceVariable):
                    raise ValueError(err_msg)
                tb = var.op.traceback[::-1]
                tb = [x for x in tb if 'tensorflow/python' not in x[0]][:5]
                raise ValueError('%s Originally defined at:\n\n%s' % (err_msg, ''.join(traceback.format_list(tb))))
            found_var = self._vars[name]
            if not shape.is_compatible_with(found_var.get_shape()):
                raise ValueError('Trying to share variable %s, but specified shape %s and found shape %s.' % (name, shape, found_var.get_shape()))
            if not dtype.is_compatible_with(found_var.dtype):
                dtype_str = dtype.name
                found_type_str = found_var.dtype.name
                raise ValueError('Trying to share variable %s, but specified dtype %s and found dtype %s.' % (name, dtype_str, found_type_str))
            return found_var
        if reuse is True:
            raise ValueError('Variable %s does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?' % name)
        if initializer is None:
            initializer, initializing_from_value = self._get_default_initializer(name=name, shape=shape, dtype=dtype)
        with ops.init_scope():
            if initializing_from_value:
                init_val = initializer
                variable_dtype = None
            else:
                if tf_inspect.isclass(initializer):
                    initializer = initializer()
                if shape.is_fully_defined():
                    if 'partition_info' in tf_inspect.getargspec(initializer).args:
                        init_val = functools.partial(initializer, shape.as_list(), dtype=dtype, partition_info=partition_info)
                    else:
                        init_val = functools.partial(initializer, shape.as_list(), dtype=dtype)
                    variable_dtype = dtype.base_dtype
                elif _needs_no_arguments(initializer):
                    init_val = initializer
                    variable_dtype = None
                else:
                    raise ValueError("The initializer passed is not valid. It should be a callable with no arguments and the shape should not be provided or an instance of `tf.keras.initializers.*' and `shape` should be fully defined.")
        if use_resource is None:
            use_resource = _DEFAULT_USE_RESOURCE
        v = _variable_v1(initial_value=init_val, name=name, trainable=trainable, collections=collections, caching_device=caching_device, dtype=variable_dtype, validate_shape=validate_shape, constraint=constraint, use_resource=use_resource, synchronization=synchronization, aggregation=aggregation)
        if context.executing_eagerly() and self._store_eager_variables:
            if collections:
                ops.add_to_collections(collections, v)
            else:
                ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, v)
            if trainable:
                ops.add_to_collection(ops.GraphKeys.TRAINABLE_VARIABLES, v)
        if not context.executing_eagerly() or self._store_eager_variables:
            self._vars[name] = v
        logging.vlog(1, 'Created variable %s with shape %s and init %s', v.name, format(shape), initializer)
        if regularizer:

            def make_regularizer_op():
                with ops.colocate_with(v):
                    with ops.name_scope(name + '/Regularizer/'):
                        return regularizer(v)
            if regularizer(v) is not None:
                lazy_eval_tensor = _LazyEvalTensor(make_regularizer_op)
                ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, lazy_eval_tensor)
        return v

    def _get_default_initializer(self, name, shape=None, dtype=dtypes.float32):
        """Provide a default initializer and a corresponding value.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.

    Returns:
      initializer and initializing_from_value. See get_variable above.

    Raises:
      ValueError: When giving unsupported dtype.
    """
        del shape
        if dtype.is_floating:
            initializer = init_ops.glorot_uniform_initializer()
            initializing_from_value = False
        elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool or (dtype == dtypes.string):
            initializer = init_ops.zeros_initializer()
            initializing_from_value = False
        else:
            raise ValueError('An initializer for variable %s of %s is required' % (name, dtype.base_dtype))
        return (initializer, initializing_from_value)