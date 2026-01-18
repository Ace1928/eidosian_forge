from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.deprecation import deprecated
class DistributedDatasetV1(input_lib.DistributedDataset):
    """Distributed dataset that supports prefetching to multiple devices."""

    def __init__(self, dataset, input_workers, strategy, num_replicas_in_sync=None, input_context=None, options=None):
        self._input_workers = input_workers
        super(DistributedDatasetV1, self).__init__(input_workers, strategy, dataset, num_replicas_in_sync=num_replicas_in_sync, input_context=input_context, options=options)

    def make_one_shot_iterator(self):
        """Get a one time use iterator for DistributedDatasetV1.

    Note: This API is deprecated. Please use `for ... in dataset:` to iterate
    over the dataset or `iter` to create an iterator.

    Returns:
      A DistributedIteratorV1 instance.
    """
        return self._make_one_shot_iterator()

    def _make_one_shot_iterator(self):
        """Get an iterator for DistributedDatasetV1."""
        if not context.executing_eagerly():
            raise ValueError('Cannot create a one shot iterator. Please use `make_initializable_iterator()` instead.')
        return self._get_iterator()

    def make_initializable_iterator(self):
        """Get an initializable iterator for DistributedDatasetV1.

    Note: This API is deprecated. Please use
    `tf.compat.v1.data.make_initializable_iterator(dataset)` to create an
    initializable iterator.

    Returns:
      A DistributedIteratorV1 instance.
    """
        return self._make_initializable_iterator()

    def _make_initializable_iterator(self, shared_name=None):
        """Get an initializable iterator for DistributedDatasetV1."""
        if context.executing_eagerly():
            raise ValueError('Cannot create initializable iterator in Eager mode. Please use `iter()` instead.')
        return self._get_iterator()

    def _get_iterator(self):
        worker_iterators = _create_iterators_per_worker(self._cloned_datasets, self._input_workers, self._options)
        cardinality = input_lib._cardinality(self._cloned_datasets[0])
        iterator = DistributedIteratorV1(self._input_workers, worker_iterators, self._strategy, cardinality, self._enable_get_next_as_optional)
        iterator._element_spec = self.element_spec
        if context.executing_eagerly():
            context.async_wait()
        return iterator

    def __iter__(self):
        if ops.executing_eagerly_outside_functions() or ops.get_default_graph().building_function:
            return self._get_iterator()
        raise RuntimeError('__iter__() is only supported inside of tf.function or when eager execution is enabled.')