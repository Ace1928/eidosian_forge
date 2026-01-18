from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.tf_export import tf_export
class _LegacyRebatchDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that divides its input batches into `num_replicas` sub-batches.

  For each batch in the input dataset, _LegacyRebatchDataset will produce
  `num_replicas` smaller batches whose sizes add up to the original batch size.

  For example:

  ```python
  ds = tf.data.Dataset.range(8)
  ds = ds.batch(4)
  ds = _LegacyRebatchDataset(ds, num_replicas=3)
  for elem in ds:
    print(elem)
  >> [0, 1], [2, 3], [], [4, 5], [6, 7], []
  ```
  """

    def __init__(self, input_dataset, num_replicas):
        """Creates a _LegacyRebatchDataset.

    Args:
      input_dataset: `Dataset` to rebatch.
      num_replicas: A `tf.int64` scalar, representing the number of sub-batches
        to split each batch from `input_dataset` into.
    """

        def recalculate_batch_size(type_spec):
            """Recalculates the output_shape after dividing it by num_replicas."""
            output_shape = type_spec._to_legacy_output_shapes()
            if not isinstance(output_shape, tensor_shape.TensorShape):
                return None
            if output_shape.rank is None:
                return None
            if len(output_shape) < 1:
                raise ValueError('Invalid `input_dataset`. Expected a dataset whose elements have rank >= 1 but found a dataset whose elements are scalars. Fix the issue by adding the `batch` transformation to the dataset.')
            output_dims = [d.value for d in output_shape.dims]
            if output_dims[0] is not None and output_dims[0] % num_replicas == 0:
                return output_dims[0] // num_replicas
            return None

        def rebatch(type_spec):
            batch_size = recalculate_batch_size(type_spec)
            return type_spec._unbatch()._batch(batch_size)
        self._element_spec = nest.map_structure(rebatch, dataset_ops.get_structure(input_dataset))
        input_dataset = dataset_ops.normalize_to_dense(input_dataset)
        variant_tensor = ged_ops.rebatch_dataset(input_dataset._variant_tensor, num_replicas=num_replicas, **self._flat_structure)
        super(_LegacyRebatchDataset, self).__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec