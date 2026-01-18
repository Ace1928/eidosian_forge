import threading
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import type_spec as type_spec_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class PerWorkerDatasetFromDatasetFunction(object):
    """Represents worker-distributed datasets created from dataset function."""

    def __init__(self, dataset_fn, coordinator):
        """Makes an iterable from datasets created by the given function.

    Args:
      dataset_fn: A function that returns a `Dataset`.
      coordinator: a `ClusterCoordinator` object, used to create dataset
        resources.
    """

        def disallow_variable_creation(next_creator, **kwargs):
            raise ValueError('Creating variables in `dataset_fn` is not allowed.')
        if isinstance(dataset_fn, def_function.Function):
            with variable_scope.variable_creator_scope(disallow_variable_creation):
                dataset_fn = dataset_fn.get_concrete_function()
        elif not isinstance(dataset_fn, tf_function.ConcreteFunction):
            with variable_scope.variable_creator_scope(disallow_variable_creation):
                dataset_fn = def_function.function(dataset_fn).get_concrete_function()
        self._dataset_fn = dataset_fn
        self._coordinator = coordinator
        self._element_spec = None

    def build(self):
        """Trigger dataset creation on workers without creating an iterator.

    Returns:
      A PerWorkerValues object containing a tuple of RemoteValues, themselves
      containing the built Dataset for each worker
    """

        def _create_per_worker_dataset():
            dataset = self._dataset_fn()
            return dataset
        per_worker_dataset = self._coordinator._create_per_worker_resources(_create_per_worker_dataset)
        dataset_fn_output_type_spec = self._dataset_fn.structured_outputs._type_spec
        for dataset_remote_value in per_worker_dataset._values:
            dataset_remote_value._type_spec = dataset_fn_output_type_spec
        return per_worker_dataset

    def __iter__(self):
        if not context.executing_eagerly() or ops.get_default_graph().building_function:
            raise RuntimeError('__iter__() is not supported inside of tf.function or in graph mode.')

        def _create_per_worker_iterator():
            dataset = self._dataset_fn()
            return iter(dataset)
        per_worker_iterator = self._coordinator._create_per_worker_resources(_create_per_worker_iterator)
        for iterator_remote_value in per_worker_iterator._values:
            iterator_remote_value._type_spec = input_lib.get_iterator_spec_from_dataset(self._coordinator.strategy, self._dataset_fn.structured_outputs)
        return PerWorkerDistributedIterator(per_worker_iterator._values)

    @property
    def element_spec(self):
        """The type specification of an element of this dataset.

    This property is subject to change without notice.
    """
        if not isinstance(self._dataset_fn, tf_function.ConcreteFunction):
            raise NotImplementedError('`element_spec` is not supported when the `dataset_fn` is not a `ConcreteFunction`.')
        return self._dataset_fn.structured_outputs.element_spec