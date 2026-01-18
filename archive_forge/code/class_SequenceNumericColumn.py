import collections
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@serialization.register_feature_column
class SequenceNumericColumn(fc.SequenceDenseColumn, collections.namedtuple('SequenceNumericColumn', ('key', 'shape', 'default_value', 'dtype', 'normalizer_fn'))):
    """Represents sequences of numeric data."""

    @property
    def _is_v2_column(self):
        return True

    @property
    def name(self):
        """See `FeatureColumn` base class."""
        return self.key

    @property
    def parse_example_spec(self):
        """See `FeatureColumn` base class."""
        return {self.key: parsing_ops.VarLenFeature(self.dtype)}

    def transform_feature(self, transformation_cache, state_manager):
        """See `FeatureColumn` base class.

    In this case, we apply the `normalizer_fn` to the input tensor.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Normalized input tensor.
    """
        input_tensor = transformation_cache.get(self.key, state_manager)
        if self.normalizer_fn is not None:
            input_tensor = self.normalizer_fn(input_tensor)
        return input_tensor

    @property
    def variable_shape(self):
        """Returns a `TensorShape` representing the shape of sequence input."""
        return tensor_shape.TensorShape(self.shape)

    def get_sequence_dense_tensor(self, transformation_cache, state_manager):
        """Returns a `TensorSequenceLengthPair`.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.
    """
        sp_tensor = transformation_cache.get(self, state_manager)
        dense_tensor = sparse_ops.sparse_tensor_to_dense(sp_tensor, default_value=self.default_value)
        dense_shape = array_ops.concat([array_ops.shape(dense_tensor)[:1], [-1], self.variable_shape], axis=0)
        dense_tensor = array_ops.reshape(dense_tensor, shape=dense_shape)
        if sp_tensor.shape.ndims == 2:
            num_elements = self.variable_shape.num_elements()
        else:
            num_elements = 1
        seq_length = fc_utils.sequence_length_from_sparse_tensor(sp_tensor, num_elements=num_elements)
        return fc.SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=seq_length)

    @property
    def parents(self):
        """See 'FeatureColumn` base class."""
        return [self.key]

    def get_config(self):
        """See 'FeatureColumn` base class."""
        config = dict(zip(self._fields, self))
        config['dtype'] = self.dtype.name
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        fc._check_config_keys(config, cls._fields)
        kwargs = fc._standardize_and_copy_config(config)
        kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
        return cls(**kwargs)