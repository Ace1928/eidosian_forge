from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class _ParseExampleDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that parses `example` dataset into a `dict` dataset."""

    def __init__(self, input_dataset, features, num_parallel_calls, deterministic):
        self._input_dataset = input_dataset
        if not structure.are_compatible(input_dataset.element_spec, tensor_spec.TensorSpec([None], dtypes.string)):
            raise TypeError(f'Input dataset should be a dataset of vectors of strings. Instead it is `{input_dataset.element_spec}`.')
        self._num_parallel_calls = num_parallel_calls
        if deterministic is None:
            self._deterministic = 'default'
        elif deterministic:
            self._deterministic = 'true'
        else:
            self._deterministic = 'false'
        self._features = parsing_ops._prepend_none_dimension(features)
        params = parsing_ops._ParseOpParams.from_features(self._features, [parsing_ops.VarLenFeature, parsing_ops.SparseFeature, parsing_ops.FixedLenFeature, parsing_ops.FixedLenSequenceFeature, parsing_ops.RaggedFeature])
        self._sparse_keys = params.sparse_keys
        self._sparse_types = params.sparse_types
        self._ragged_keys = params.ragged_keys
        self._ragged_value_types = params.ragged_value_types
        self._ragged_split_types = params.ragged_split_types
        self._dense_keys = params.dense_keys
        self._dense_defaults = params.dense_defaults_vec
        self._dense_shapes = params.dense_shapes_as_proto
        self._dense_types = params.dense_types
        input_dataset_shape = dataset_ops.get_legacy_output_shapes(self._input_dataset)
        self._element_spec = {}
        for key, value_type in zip(params.sparse_keys, params.sparse_types):
            self._element_spec[key] = sparse_tensor.SparseTensorSpec(input_dataset_shape.concatenate([None]), value_type)
        for key, value_type, dense_shape in zip(params.dense_keys, params.dense_types, params.dense_shapes):
            self._element_spec[key] = tensor_spec.TensorSpec(input_dataset_shape.concatenate(dense_shape), value_type)
        for key, value_type, splits_type in zip(params.ragged_keys, params.ragged_value_types, params.ragged_split_types):
            self._element_spec[key] = ragged_tensor.RaggedTensorSpec(input_dataset_shape.concatenate([None]), value_type, 1, splits_type)
        variant_tensor = gen_experimental_dataset_ops.parse_example_dataset_v2(self._input_dataset._variant_tensor, self._num_parallel_calls, self._dense_defaults, self._sparse_keys, self._dense_keys, self._sparse_types, self._dense_shapes, deterministic=self._deterministic, ragged_keys=self._ragged_keys, ragged_value_types=self._ragged_value_types, ragged_split_types=self._ragged_split_types, **self._flat_structure)
        super(_ParseExampleDataset, self).__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec