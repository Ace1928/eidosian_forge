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
class DistributedDataset(_IterableInput, composite_tensor.CompositeTensor):
    """Distributed dataset that supports prefetching to multiple devices."""

    def __init__(self, input_workers, strategy, dataset=None, num_replicas_in_sync=None, input_context=None, components=None, element_spec=None, enable_get_next_as_optional=None, build=True, options=None, replica_order=None):
        """Distribute the dataset on all workers.

    If `num_replicas_in_sync` is not None, we split each batch of the dataset
    into `num_replicas_in_sync` smaller batches, to be distributed among that
    worker's replicas, so that the batch size for a global step (across all
    workers and replicas) is as expected.

    Args:
      input_workers: an `InputWorkers` object.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
      dataset: `tf.data.Dataset` that will be used as the input source. Either
        dataset or components field should be passed when constructing
        DistributedDataset. Use this when contructing DistributedDataset from a
        new `tf.data.Dataset`. Use components when constructing using
        DistributedDatasetSpec.
      num_replicas_in_sync: Optional integer. If this is not None, the value is
        used to decide how to rebatch datasets into smaller batches so that the
        total batch size for each step (across all workers and replicas) adds up
        to `dataset`'s batch size.
      input_context: `InputContext` for sharding. Only pass this in for between
        graph multi-worker cases where there is only one `input_worker`. In
        these cases, we will shard based on the `input_pipeline_id` and
        `num_input_pipelines` in the `InputContext`.
      components: datasets when DistributedDataset is constructed from
        DistributedDatasetSpec. Either field dataset or components should be
        passed.
      element_spec: element spec for DistributedDataset when constructing from
        DistributedDatasetSpec. This will be used to set the element_spec for
        DistributedDataset and verified against element_spec from components.
      enable_get_next_as_optional: this is required when components is passed
        instead of dataset.
      build: whether to build underlying datasets when this object is created.
        This is only useful for `ParameterServerStrategy` now.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.
      replica_order: the order of the replicas, which will be used to reorder
        the iterators to match the device order.
    """
        super(DistributedDataset, self).__init__(input_workers=input_workers)
        if input_workers is None or strategy is None:
            raise ValueError('input_workers and strategy are required arguments')
        if dataset is not None and components is not None:
            raise ValueError('Only one of dataset or components should be present')
        if dataset is None and components is None:
            raise ValueError('At least one of dataset or components should be passed')
        self._input_workers = input_workers
        self._strategy = strategy
        self._options = options
        self._input_context = input_context
        self._num_replicas_in_sync = num_replicas_in_sync
        self._replica_order = replica_order
        if dataset is not None:
            self._original_dataset = dataset
            self._built = False
            if build:
                self.build()
        else:
            if not build:
                raise ValueError('When constructing DistributedDataset with components, build should not be False. This is an internal error. Please file a bug.')
            if enable_get_next_as_optional is None:
                raise ValueError('When constructing DistributedDataset with components, ' + 'enable_get_next_as_optional should also be passed')
            self._cloned_datasets = components
            self._cardinality = _cardinality(self._cloned_datasets[0])
            self._enable_get_next_as_optional = enable_get_next_as_optional
            assert element_spec is not None
            if element_spec != _create_distributed_tensor_spec(self._strategy, self._cloned_datasets[0].element_spec):
                raise ValueError('Mismatched element_spec from the passed components')
            self._element_spec = element_spec
            self._built = True

    def build(self, dataset_to_replace=None):
        assert not self._built
        dataset = dataset_to_replace or self._original_dataset
        self._cardinality = _cardinality(dataset)
        self._enable_get_next_as_optional = _enable_get_next_as_optional(self._strategy, dataset, self._cardinality)
        distribute_start_time_ns = time.time_ns()
        self._create_cloned_datasets_from_dataset(dataset, self._input_context, self._input_workers, self._strategy, self._num_replicas_in_sync)
        if context.executing_eagerly():
            context.async_wait()
            distribute_duration_ms = (time.time_ns() - distribute_start_time_ns) // 1000000
            _distributed_dataset_initialization_time_milliseconds.get_cell(self._strategy.__class__.__name__, str(self._input_workers.num_workers)).add(distribute_duration_ms)
        self._element_spec = _create_distributed_tensor_spec(self._strategy, self._cloned_datasets[0].element_spec)
        self._built = True

    def auto_shard(self, num_shards, shard_ix):
        assert len(self._cloned_datasets) == len(self._input_workers.worker_devices), f'datasets: {len(self._cloned_datasets)}, input workers: {len(self._input_workers.worker_devices)}'
        sharded_datasets = []
        for i in range(len(self._input_workers.worker_devices)):
            with ops.colocate_with(self._cloned_datasets[i]._variant_tensor):
                sharded_datasets.append(input_ops.auto_shard_dataset(self._cloned_datasets[i], num_shards, shard_ix, self._num_replicas_in_sync))
        return DistributedDataset(self._input_workers, self._strategy, components=sharded_datasets, element_spec=self._element_spec, options=self._options, enable_get_next_as_optional=self._enable_get_next_as_optional)

    @property
    def cardinality(self):
        if not self._built:
            raise ValueError('Cannot get the cardinality of a dataset that is not built')
        return self._cardinality

    def _create_cloned_datasets_from_dataset(self, dataset, input_context, input_workers, strategy, num_replicas_in_sync):
        if num_replicas_in_sync is not None and num_replicas_in_sync > 1:
            num_workers = input_context.num_input_pipelines if input_context else len(input_workers.worker_devices)
            rebatch_fn = self._make_rebatch_fn(dataset, num_workers, num_replicas_in_sync)
        else:
            rebatch_fn = None
        self._cloned_datasets = []
        if input_context:
            assert input_workers.num_workers == 1
            if rebatch_fn is not None:
                dataset = rebatch_fn(dataset, input_context.input_pipeline_id)
            dataset = input_ops.auto_shard_dataset(dataset, input_context.num_input_pipelines, input_context.input_pipeline_id, num_replicas_in_sync)
            self._cloned_datasets.append(dataset)
        else:
            replicated_ds = distribute.replicate(dataset, input_workers.worker_devices)
            for i, worker in enumerate(input_workers.worker_devices):
                with ops.device(worker):
                    cloned_dataset = replicated_ds[worker]
                    if rebatch_fn is not None:
                        cloned_dataset = rebatch_fn(cloned_dataset, i)
                    cloned_dataset = input_ops.auto_shard_dataset(cloned_dataset, len(input_workers.worker_devices), i, num_replicas_in_sync)
                    self._cloned_datasets.append(cloned_dataset)

    def _make_rebatch_fn(self, dataset, num_workers, num_replicas_in_sync):
        """Returns a callable that rebatches the input dataset.

    Args:
      dataset: A `tf.data.Dataset` representing the dataset to be distributed.
      num_workers: An integer representing the number of workers to distribute
        `dataset` among.
      num_replicas_in_sync: An integer representing the number of replicas in
        sync across all workers.
    """
        if num_replicas_in_sync % num_workers:
            raise ValueError('tf.distribute expects every worker to have the same number of replicas. However, encountered `num_replicas_in_sync` ({}) that cannot be divided by `num_workers` ({})'.format(num_replicas_in_sync, num_workers))
        num_replicas_per_worker = num_replicas_in_sync // num_workers
        with ops.colocate_with(dataset._variant_tensor):
            batch_size = distribute.compute_batch_size(dataset)

        def rebatch_fn(dataset, worker_index):
            try:

                def apply_rebatch():
                    batch_sizes = distribute.batch_sizes_for_worker(batch_size, num_workers, num_replicas_per_worker, worker_index)
                    return dataset.rebatch(batch_sizes).prefetch(num_replicas_per_worker)

                def apply_legacy_rebatch():
                    return distribute._LegacyRebatchDataset(dataset, num_replicas_in_sync).prefetch(num_replicas_per_worker)
                with ops.colocate_with(dataset._variant_tensor):
                    return tf_cond.cond(math_ops.not_equal(batch_size, -1), true_fn=apply_rebatch, false_fn=apply_legacy_rebatch)
            except errors.InvalidArgumentError as e:
                if 'without encountering a batch' in str(e):
                    six.reraise(ValueError, ValueError('Call the `batch` method on the input Dataset in order to be able to split your input across {} replicas.\n Please see the tf.distribute.Strategy guide. {}'.format(num_replicas_in_sync, e)), sys.exc_info()[2])
                else:
                    raise
        return rebatch_fn

    def __iter__(self):
        if not (context.executing_eagerly() or ops.get_default_graph().building_function):
            raise RuntimeError('__iter__() is only supported inside of tf.function or when eager execution is enabled.')
        if not self._built:
            raise ValueError('To use this dataset, you need to pass this dataset to ClusterCoordinator.create_per_worker_dataset.')
        canonicalize_devices = getattr(self._strategy, '_canonicalize_devices', True)
        worker_iterators = _create_iterators_per_worker(self._cloned_datasets, self._input_workers, options=self._options, canonicalize_devices=canonicalize_devices)
        iterator = DistributedIterator(self._input_workers, worker_iterators, self._strategy, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional, options=self._options, replica_order=self._replica_order)
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
        return DistributedDatasetSpec(self._input_workers, self._element_spec, self._strategy, self._options, enable_get_next_as_optional=self._enable_get_next_as_optional)