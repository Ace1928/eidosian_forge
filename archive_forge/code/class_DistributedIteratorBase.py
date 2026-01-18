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
class DistributedIteratorBase(collections_abc.Iterator, distribute_types.DistributedIteratorInterface):
    """Common implementation for all input iterators."""

    def __init__(self, input_workers, iterators, strategy, cardinality, enable_get_next_as_optional, replica_order=None):
        assert isinstance(input_workers, InputWorkers)
        if not input_workers.worker_devices:
            raise ValueError('Should have at least one worker for input iterator.')
        self._iterators = iterators
        self._input_workers = input_workers
        self._strategy = strategy
        self._cardinality = cardinality
        self._enable_get_next_as_optional = enable_get_next_as_optional
        self._replica_order = replica_order

    def next(self):
        return self.__next__()

    def __next__(self):
        try:
            return self.get_next()
        except errors.OutOfRangeError:
            raise StopIteration

    def __iter__(self):
        return self

    def get_next_as_optional(self):
        if self._cardinality == cardinality_lib.INFINITE:
            return optional_ops.Optional.from_value(self._get_next_no_partial_batch_handling())
        if self._cardinality == 0 and (not self._strategy.extended._in_multi_worker_mode()):
            return optional_ops.Optional.empty(self._element_spec)
        optional_list = []
        for i, worker in enumerate(self._input_workers.worker_devices):
            with ops.device(worker):
                optional_list.append(self._iterators[i].get_next_as_optional_list())

        def _create_optional_with_dummy():
            value_list = _get_value_or_dummy(self._input_workers, optional_list, produce_dummy=True)
            if self._replica_order is not None:
                value_list = self._reorder_replicas(value_list)
            per_replica = _create_per_replica(value_list, self._strategy)
            return optional_ops.Optional.from_value(per_replica)

        def _create_empty_optional():
            return optional_ops.Optional.empty(self._element_spec)
        num_replicas_with_values = _calculate_replicas_with_values(self._strategy, self._input_workers, optional_list)
        return tf_cond.cond(num_replicas_with_values > 0, _create_optional_with_dummy, _create_empty_optional, strict=True)

    def get_next(self, name=None):
        """Returns the next input from the iterator for all replicas."""
        with distribute_lib.enter_or_assert_strategy(self._strategy):
            if distribute_lib.get_replica_context() is not None:
                raise ValueError('next(iterator) should be called from outside of replica_fn. e.g. strategy.run(replica_fn, args=(next(iterator),))')
        if not self._enable_get_next_as_optional:
            return self._get_next_no_partial_batch_handling(name)
        optional_list = []
        for i, worker in enumerate(self._input_workers.worker_devices):
            with ops.device(worker):
                optional_list.append(self._iterators[i].get_next_as_optional_list())
        num_replicas_with_values = _calculate_replicas_with_values(self._strategy, self._input_workers, optional_list)

        def _value_or_dummy():
            value_list = _get_value_or_dummy(self._input_workers, optional_list, produce_dummy=True)
            if self._replica_order is not None:
                value_list = self._reorder_replicas(value_list)
            return _create_per_replica(value_list, self._strategy)

        def _eof():
            return self._get_next_no_partial_batch_handling()
        return tf_cond.cond(num_replicas_with_values > 0, _value_or_dummy, _eof, strict=True)

    def _get_next_no_partial_batch_handling(self, name=None):
        replicas = []
        for i, worker in enumerate(self._input_workers.worker_devices):
            if name is not None:
                d = tf_device.DeviceSpec.from_string(worker)
                new_name = '%s_%s_%d' % (name, d.job, d.task)
            else:
                new_name = None
            with ops.device(worker):
                replicas.extend(self._iterators[i].get_next_as_list(new_name))
        if self._replica_order is not None:
            replicas = self._reorder_replicas(replicas)
        return _create_per_replica(replicas, self._strategy)

    def _reorder_replicas(self, replicas):
        assert len(self._replica_order) == len(replicas), 'replica order size ({}) != replicas size ({})!'.format(len(self._replica_order), len(replicas))
        return [replicas[i] for i in self._replica_order]