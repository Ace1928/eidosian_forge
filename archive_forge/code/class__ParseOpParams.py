import collections
import re
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import tf_export
class _ParseOpParams:
    """Raw parameters used by `gen_parsing_ops`.

  Attributes:
    sparse_keys: A list of string keys in the examples' features. The results
      for these keys will be returned as `SparseTensor` objects.
    sparse_types: A list of `DTypes` of the same length as `sparse_keys`. Only
      `tf.float32` (`FloatList`), `tf.int64` (`Int64List`), and `tf.string`
      (`BytesList`) are supported.
    dense_keys: A list of string keys in the examples' features. The results for
      these keys will be returned as `Tensor`s
    dense_types: A list of DTypes of the same length as `dense_keys`. Only
      `tf.float32` (`FloatList`), `tf.int64` (`Int64List`), and `tf.string`
      (`BytesList`) are supported.
    dense_defaults: A dict mapping string keys to `Tensor`s. The keys of the
      dict must match the dense_keys of the feature.
    dense_shapes: A list of tuples with the same length as `dense_keys`. The
      shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys`.  Must be either
      fully defined, or may contain an unknown first dimension. An unknown first
      dimension means the feature is treated as having a variable number of
      blocks, and the output shape along this dimension is considered unknown at
      graph build time.  Padding is applied for minibatch elements smaller than
      the maximum number of blocks for the given feature along this dimension.
    ragged_keys: A list of string keys in the examples' features.  The
      results for these keys will be returned as `RaggedTensor` objects.
    ragged_value_types: A list of `DTypes` of the same length as `ragged_keys`,
      specifying the value type for each ragged feature.  Must be one of:
      `tf.float32`, `tf.int64`, `tf.string`.
    ragged_split_types: A list of `DTypes` of the same length as `ragged_keys`,
      specifying the row_splits type for each ragged feature.  Must be one of:
      `tf.int32`, `tf.int64`.
    dense_shapes_as_proto: dense_shapes converted to TensorShapeProto.
    dense_defaults_vec: A vector of `Tensor`s containing the default values,
      corresponding 1:1 with `dense_keys`.
    num_features: The total number of feature keys.
  """

    def __init__(self, sparse_keys=None, sparse_types=None, dense_keys=None, dense_types=None, dense_defaults=None, dense_shapes=None, ragged_keys=None, ragged_value_types=None, ragged_split_types=None):
        dense_defaults = collections.OrderedDict() if dense_defaults is None else dense_defaults
        sparse_keys = [] if sparse_keys is None else sparse_keys
        sparse_types = [] if sparse_types is None else sparse_types
        dense_keys = [] if dense_keys is None else dense_keys
        dense_types = [] if dense_types is None else dense_types
        dense_shapes = [[]] * len(dense_keys) if dense_shapes is None else dense_shapes
        ragged_keys = [] if ragged_keys is None else ragged_keys
        ragged_value_types = [] if ragged_value_types is None else ragged_value_types
        ragged_split_types = [] if ragged_split_types is None else ragged_split_types
        self.sparse_keys = sparse_keys
        self.sparse_types = [dtypes.as_dtype(t) for t in sparse_types]
        self.dense_keys = dense_keys
        self.dense_types = [dtypes.as_dtype(t) for t in dense_types]
        self.dense_shapes = [tensor_shape.as_shape(s) for s in dense_shapes]
        self.dense_defaults = dense_defaults
        self.ragged_keys = ragged_keys
        self.ragged_value_types = [dtypes.as_dtype(t) for t in ragged_value_types]
        self.ragged_split_types = [dtypes.as_dtype(t) for t in ragged_split_types]
        self._validate()

    @classmethod
    def from_features(cls, features, types):
        """Builds _ParseOpParams for a given set of features and allowed types.

    Args:
      features: A `dict` mapping feature keys to objects of a type in `types`.
      types: Type of features to allow, among `FixedLenFeature`,
        `VarLenFeature`, `SparseFeature`, and `FixedLenSequenceFeature`.

    Returns:
      A `_ParseOpParams` containing the raw parameters for `gen_parsing_ops`.

    Raises:
      ValueError: if `features` contains an item not in `types`, or an invalid
          feature.
      ValueError: if sparse and dense key sets intersect.
      ValueError: if input lengths do not match up.
    """
        params = cls()
        if features:
            for key in sorted(features.keys()):
                feature = features[key]
                if not isinstance(feature, tuple(types)):
                    raise ValueError(f"Unsupported {type(feature).__name__} {feature} for key '{key}'")
                params._add_feature(key, feature)
        params._validate()
        return params

    @property
    def dense_shapes_as_proto(self):
        return [shape.as_proto() for shape in self.dense_shapes]

    @property
    def num_features(self):
        return len(self.dense_keys) + len(self.sparse_keys) + len(self.ragged_keys)

    @property
    def dense_defaults_vec(self):
        return [self._make_dense_default(k, s, t) for k, s, t in zip(self.dense_keys, self.dense_shapes, self.dense_types)]

    def _make_dense_default(self, key, shape, dtype):
        """Construct the default value tensor for a specified dense feature.

    Args:
      key: The key string identifying the dense feature.
      shape: The dense feature's shape.
      dtype: The dense feature's dtype.

    Returns:
      A Tensor.
    """
        default_value = self.dense_defaults.get(key)
        if shape.ndims is not None and shape.ndims > 0 and (shape.dims[0].value is None):
            if default_value is None:
                default_value = ops.convert_to_tensor('' if dtype == dtypes.string else 0, dtype=dtype)
            else:
                key_name = 'padding_' + re.sub('[^A-Za-z0-9_.\\-/]', '_', key)
                default_value = ops.convert_to_tensor(default_value, dtype=dtype, name=key_name)
                default_value = array_ops.reshape(default_value, [])
        elif default_value is None:
            default_value = constant_op.constant([], dtype=dtype)
        elif not isinstance(default_value, tensor.Tensor):
            key_name = 'key_' + re.sub('[^A-Za-z0-9_.\\-/]', '_', key)
            default_value = ops.convert_to_tensor(default_value, dtype=dtype, name=key_name)
            default_value = array_ops.reshape(default_value, shape)
        return default_value

    def _add_feature(self, key, feature):
        """Adds the specified feature to this ParseOpParams."""
        if isinstance(feature, VarLenFeature):
            self._add_varlen_feature(key, feature)
        elif isinstance(feature, SparseFeature):
            self._add_sparse_feature(key, feature)
        elif isinstance(feature, FixedLenFeature):
            self._add_fixed_len_feature(key, feature)
        elif isinstance(feature, FixedLenSequenceFeature):
            self._add_fixed_len_sequence_feature(key, feature)
        elif isinstance(feature, RaggedFeature):
            self._add_ragged_feature(key, feature)
        else:
            raise ValueError(f'Invalid feature {key}:{feature}.')

    def _add_varlen_feature(self, key, feature):
        """Adds a VarLenFeature."""
        if not feature.dtype:
            raise ValueError(f'Missing type for feature {key}. Received feature={feature}')
        self._add_sparse_key(key, feature.dtype)

    def _add_sparse_key(self, key, dtype):
        """Adds a sparse key & dtype, checking for duplicates."""
        if key in self.sparse_keys:
            original_dtype = self.sparse_types[self.sparse_keys.index(key)]
            if original_dtype != dtype:
                raise ValueError(f'Conflicting type {original_dtype} vs {dtype} for feature {key}.')
        else:
            self.sparse_keys.append(key)
            self.sparse_types.append(dtype)

    def _add_sparse_feature(self, key, feature):
        """Adds a SparseFeature."""
        if not feature.index_key:
            raise ValueError(f'Missing index_key for SparseFeature {feature}.')
        if not feature.value_key:
            raise ValueError(f'Missing value_key for SparseFeature {feature}.')
        if not feature.dtype:
            raise ValueError(f'Missing type for feature {key}. Received feature={feature}.')
        index_keys = feature.index_key
        if isinstance(index_keys, str):
            index_keys = [index_keys]
        elif len(index_keys) > 1:
            tf_logging.warning('SparseFeature is a complicated feature config and should only be used after careful consideration of VarLenFeature.')
        for index_key in sorted(index_keys):
            self._add_sparse_key(index_key, dtypes.int64)
        self._add_sparse_key(feature.value_key, feature.dtype)

    def _add_fixed_len_feature(self, key, feature):
        """Adds a FixedLenFeature."""
        if not feature.dtype:
            raise ValueError(f'Missing type for feature {key}. Received feature={feature}.')
        if feature.shape is None:
            raise ValueError(f'Missing shape for feature {key}. Received feature={feature}.')
        feature_tensor_shape = tensor_shape.as_shape(feature.shape)
        if feature.shape and feature_tensor_shape.ndims and (feature_tensor_shape.dims[0].value is None):
            raise ValueError(f'First dimension of shape for feature {key} unknown. Consider using FixedLenSequenceFeature. Received feature={feature}.')
        if feature.shape is not None and (not feature_tensor_shape.is_fully_defined()):
            raise ValueError(f'All dimensions of shape for feature {key} need to be known but received {feature.shape!s}.')
        self.dense_keys.append(key)
        self.dense_shapes.append(tensor_shape.as_shape(feature.shape))
        self.dense_types.append(feature.dtype)
        if feature.default_value is not None:
            self.dense_defaults[key] = feature.default_value

    def _add_fixed_len_sequence_feature(self, key, feature):
        """Adds a FixedLenSequenceFeature."""
        if not feature.dtype:
            raise ValueError(f'Missing type for feature {key}. Received feature={feature}.')
        if feature.shape is None:
            raise ValueError(f'Missing shape for feature {key}. Received feature={feature}.')
        self.dense_keys.append(key)
        self.dense_shapes.append(tensor_shape.as_shape(feature.shape))
        self.dense_types.append(feature.dtype)
        if feature.allow_missing:
            self.dense_defaults[key] = None
        if feature.default_value is not None:
            self.dense_defaults[key] = feature.default_value

    def _add_ragged_key(self, key, value_type, split_type):
        """Adds a ragged key & dtype, checking for duplicates."""
        if key in self.ragged_keys:
            original_value_type = self.ragged_value_types[self.ragged_keys.index(key)]
            original_split_type = self.ragged_split_types[self.ragged_keys.index(key)]
            if original_value_type != value_type:
                raise ValueError(f'Conflicting type {original_value_type} vs {value_type} for feature {key}.')
            if original_split_type != split_type:
                raise ValueError(f'Conflicting partition type {original_split_type} vs {split_type} for feature {key}.')
        else:
            self.ragged_keys.append(key)
            self.ragged_value_types.append(value_type)
            self.ragged_split_types.append(split_type)

    def _add_ragged_feature(self, key, feature):
        """Adds a RaggedFeature."""
        value_key = key if feature.value_key is None else feature.value_key
        self._add_ragged_key(value_key, feature.dtype, feature.row_splits_dtype)
        for partition in feature.partitions:
            if not isinstance(partition, RaggedFeature.UniformRowLength):
                self._add_ragged_key(partition.key, dtypes.int64, feature.row_splits_dtype)

    def _validate(self):
        """Validates the features in this ParseOpParams."""
        if len(self.dense_shapes) != len(self.dense_keys):
            raise ValueError(f'len(self.dense_shapes) != len(self.dense_keys): {len(self.dense_shapes)} vs {len(self.dense_keys)}.')
        if len(self.dense_types) != len(self.dense_keys):
            raise ValueError(f'len(self.dense_types) != len(self.dense_keys): {len(self.dense_types)} vs {len(self.dense_keys)}.')
        if len(self.sparse_types) != len(self.sparse_keys):
            raise ValueError(f'len(self.sparse_types) != len(self.sparse_keys): {len(self.sparse_types)} vs {len(self.sparse_keys)}.')
        if len(self.ragged_value_types) != len(self.ragged_keys):
            raise ValueError(f'len(self.ragged_value_types) != len(self.ragged_keys): {len(self.ragged_value_types)} vs {len(self.ragged_keys)}.')
        if len(self.ragged_split_types) != len(self.ragged_keys):
            raise ValueError(f'len(self.ragged_split_types) != len(self.ragged_keys): {len(self.ragged_split_types)} vs {len(self.ragged_keys)}.')
        dense_key_set = set(self.dense_keys)
        sparse_key_set = set(self.sparse_keys)
        ragged_key_set = set(self.ragged_keys)
        if not dense_key_set.isdisjoint(sparse_key_set):
            raise ValueError(f'Dense and sparse keys must not intersect; dense_keys: {self.dense_keys}, sparse_keys: {self.sparse_keys}, intersection: {dense_key_set.intersection(sparse_key_set)}')
        if not dense_key_set.isdisjoint(ragged_key_set):
            raise ValueError('Dense and ragged keys must not intersect; dense_keys: ', f'{self.dense_keys}, ragged_keys: {self.ragged_keys}, intersection: {dense_key_set.intersection(ragged_key_set)}')
        if not ragged_key_set.isdisjoint(sparse_key_set):
            raise ValueError(f'Ragged and sparse keys must not intersect; ragged_keys: {self.ragged_keys}, sparse_keys: {self.sparse_keys}, intersection: {ragged_key_set.intersection(sparse_key_set)}')