import functools
import sys
import time
import six
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
class DistributedDatasetsFromFunction(_IterableInput, composite_tensor.CompositeTensor):
    """Inputs created from dataset function."""

    def __init__(self, input_workers, strategy, input_contexts=None, dataset_fn=None, options=None, components=None, element_spec=None, build=True, replica_order=None):
        """Makes an iterable from datasets created by the given function.

    Args:
      input_workers: an `InputWorkers` object.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
      input_contexts: A list of `InputContext` instances to be passed to call(s)
        to `dataset_fn`. Length and order should match worker order in
        `worker_device_pairs`.
      dataset_fn: A function that returns a `Dataset` given an `InputContext`.
        Either dataset_fn or components should be passed to construct
        DistributedDatasetsFromFunction. Use this when constructing
        DistributedDataset using a function. Use components when constructing
        using DistributedDatasetsFromFunctionSpec.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.
      components: datasets when DistributedDatasetsFromFunction is constructed
        from DistributedDatasetsFromFunctionSpec. Only one of dataset or
        components should be passed.
      element_spec: element spec for DistributedDataset when constructing from
        DistributedDatasetSpec. This will be used to set the element_spec for
        DistributedDatasetsFromFunctionSpec and verified against element_spec
        from components.
      build: whether to build underlying datasets when this object is created.
        This is only useful for `ParameterServerStrategy` now.
      replica_order: the order of the replicas, which will be used to reorder
        the iterators to match the device order.
    """
        super(DistributedDatasetsFromFunction, self).__init__(input_workers=input_workers)
        self._input_workers = input_workers
        self._strategy = strategy
        self._options = options
        self._replica_order = replica_order
        if dataset_fn is not None and components is not None:
            raise ValueError('Only one of dataset_fn or components should be set')
        if dataset_fn is None and components is None:
            raise ValueError('At least one of dataset_fn or components should be set')
        if dataset_fn is not None:
            if input_workers.num_workers != len(input_contexts):
                raise ValueError('Number of input workers (%d) is not same as number of input_contexts (%d)' % (input_workers.num_workers, len(input_contexts)))
            self._input_contexts = input_contexts
            self._num_replicas_in_sync = self._input_contexts[0].num_replicas_in_sync
            self._dataset_fn = dataset_fn
            self._built = False
            if build:
                self.build()
        else:
            if element_spec is None:
                raise ValueError('element_spec should also be passed when passing components')
            if not build:
                raise ValueError('When constructing DistributedDatasetFromFunction with components, build should not be False. This is an internal error. Please file a bug.')
            self._element_spec = element_spec
            self._datasets = components
            self._num_replicas_in_sync = None
            self._built = True
            self._cardinality = _cardinality(self._datasets[0])
            self._enable_get_next_as_optional = _enable_get_next_as_optional(self._strategy, self._datasets[0], self._cardinality)

    def build(self):
        assert not self._built
        distribute_start_time_ns = time.time_ns()
        self._datasets, element_spec = _create_datasets_from_function_with_input_context(self._input_contexts, self._input_workers, self._dataset_fn)
        if context.executing_eagerly():
            context.async_wait()
            distribute_duration_ms = (time.time_ns() - distribute_start_time_ns) // 1000000
            _distributed_dataset_from_function_initialization_time_milliseconds.get_cell(self._strategy.__class__.__name__, str(self._input_workers.num_workers)).add(distribute_duration_ms)
        self._element_spec = _create_distributed_tensor_spec(self._strategy, element_spec)
        self._cardinality = _cardinality(self._datasets[0])
        self._enable_get_next_as_optional = _enable_get_next_as_optional(self._strategy, self._datasets[0], self._cardinality)
        self._built = True

    def auto_shard(self, num_shards, shard_ix):
        assert len(self._datasets) == len(self._input_workers.worker_devices), f'datasets: {len(self._datasets)}, input workers: {len(self._input_workers.worker_devices)}'
        sharded_datasets = []
        for i in range(len(self._input_workers.worker_devices)):
            with ops.colocate_with(self._datasets[i]._variant_tensor):
                sharded_datasets.append(input_ops.auto_shard_dataset(self._datasets[i], num_shards, shard_ix, self._num_replicas_in_sync))
        return DistributedDatasetsFromFunction(self._input_workers, self._strategy, components=sharded_datasets, element_spec=self._element_spec, options=self._options)

    @property
    def cardinality(self):
        if not self._built:
            raise ValueError('Cannot get the cardinality of a dataset that is not built')
        return self._cardinality

    def __iter__(self):
        if not (ops.executing_eagerly_outside_functions() or ops.get_default_graph().building_function):
            raise RuntimeError('__iter__() is only supported inside of tf.function or when eager execution is enabled.')
        if not self._built:
            raise ValueError('You need to use this dataset in ClusterCoordinator.create_per_worker_dataset.')
        canonicalize_devices = getattr(self._strategy, '_canonicalize_devices', True)
        iterators = _create_iterators_per_worker(self._datasets, self._input_workers, options=self._options, canonicalize_devices=canonicalize_devices)
        iterator = DistributedIterator(input_workers=self._input_workers, iterators=iterators, strategy=self._strategy, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional, options=self._options, replica_order=self._replica_order)
        iterator._element_spec = self._element_spec
        if context.executing_eagerly():
            context.async_wait()
        return iterator

    @property
    def element_spec(self):
        """The type specification of an element of this dataset."""
        if self._enable_get_next_as_optional and self._strategy.extended._in_multi_worker_mode():
            return nest.map_structure(_rebatch_as_dynamic, self._element_spec, expand_composites=False)
        return self._element_spec

    @property
    def _type_spec(self):
        return DistributedDatasetsFromFunctionSpec(self._input_workers, self._element_spec, self._strategy, self._options)