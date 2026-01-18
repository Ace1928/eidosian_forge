import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops
class _PaddedBatchDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that batches and pads contiguous elements from its input."""

    def __init__(self, input_dataset, batch_size, padded_shapes, padding_values, drop_remainder, name=None):
        """See `Dataset.batch()` for details."""
        self._input_dataset = input_dataset

        def check_types(component_spec):
            if not isinstance(component_spec, tensor_spec.TensorSpec):
                if isinstance(component_spec, dataset_ops.DatasetSpec):
                    raise TypeError('`padded_batch` is not supported for datasets of datasets')
                raise TypeError(f'`padded_batch` is only supported for datasets that produce tensor elements but type spec of elements in the input dataset is not a subclass of TensorSpec: `{component_spec}`.')
        nest.map_structure(check_types, input_dataset.element_spec)
        self._input_dataset = input_dataset
        self._batch_size = ops.convert_to_tensor(batch_size, dtype=dtypes.int64, name='batch_size')
        padding_values = _padding_values_or_default(padding_values, input_dataset)
        input_shapes = dataset_ops.get_legacy_output_shapes(input_dataset)
        flat_padded_shapes = nest.flatten_up_to(input_shapes, padded_shapes)
        flat_padded_shapes_as_tensors = []
        for input_component_shape, padded_shape in zip(nest.flatten(input_shapes), flat_padded_shapes):
            flat_padded_shapes_as_tensors.append(_padded_shape_to_tensor(padded_shape, input_component_shape))
        self._padded_shapes = nest.pack_sequence_as(input_shapes, flat_padded_shapes_as_tensors)
        if nest.is_nested(input_shapes) and (not nest.is_nested(padding_values)):
            padding_values = nest.map_structure(lambda _: padding_values, input_shapes)
        self._padding_values = nest.map_structure_up_to(input_shapes, _padding_value_to_tensor, padding_values, dataset_ops.get_legacy_output_types(input_dataset))
        self._drop_remainder = ops.convert_to_tensor(drop_remainder, dtype=dtypes.bool, name='drop_remainder')

        def _padded_shape_to_batch_shape(s):
            return tensor_shape.TensorShape([tensor_util.constant_value(self._batch_size) if smart_cond.smart_constant_value(self._drop_remainder) else None]).concatenate(tensor_util.constant_value_as_shape(s))
        output_shapes = nest.map_structure(_padded_shape_to_batch_shape, self._padded_shapes)
        self._structure = structure.convert_legacy_structure(dataset_ops.get_legacy_output_types(self._input_dataset), output_shapes, dataset_ops.get_legacy_output_classes(self._input_dataset))
        self._name = name
        variant_tensor = gen_dataset_ops.padded_batch_dataset_v2(input_dataset._variant_tensor, batch_size=self._batch_size, padded_shapes=[ops.convert_to_tensor(s, dtype=dtypes.int64) for s in nest.flatten(self._padded_shapes)], padding_values=nest.flatten(self._padding_values), drop_remainder=self._drop_remainder, output_shapes=structure.get_flat_tensor_shapes(self._structure), metadata=self._metadata.SerializeToString())
        super().__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        return self._structure