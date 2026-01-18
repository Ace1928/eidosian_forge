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
class PerWorkerDatasetFromDataset(PerWorkerDatasetFromDatasetFunction):
    """Represents worker-distributed datasets created from a dataset."""

    def __init__(self, dataset, coordinator):
        """Makes an iterable from datasets created by the given dataset.

    It creates a dataset_fn which deserializes a dataset from a graph under the
    hood.

    Args:
      dataset: A tf.data.Dataset, a DistributedDataset or a
        DistributedDatasetsFromFunction
      coordinator: a `ClusterCoordinator` object, used to create dataset
        resources.
    """
        if isinstance(dataset, input_lib.DistributedDataset):
            original_dataset = dataset._original_dataset
            serialized = serialize_dataset_to_graph(original_dataset)

            def dataset_fn():
                deserialized = deserialize_dataset_from_graph(serialized, original_dataset.element_spec)
                dataset.build(dataset_to_replace=deserialized)
                return dataset
        elif isinstance(dataset, input_lib.DistributedDatasetsFromFunction):

            def dataset_fn():
                dataset.build()
                return dataset
        elif isinstance(dataset, dataset_ops.Dataset):
            serialized = serialize_dataset_to_graph(dataset)

            def dataset_fn():
                return deserialize_dataset_from_graph(serialized, dataset.element_spec)
        else:
            raise ValueError('Unexpected dataset type!')
        super(PerWorkerDatasetFromDataset, self).__init__(dataset_fn, coordinator)