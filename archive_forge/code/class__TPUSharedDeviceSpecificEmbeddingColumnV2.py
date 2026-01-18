import copy
import enum
import math
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.feature_column import _is_running_on_cpu
from tensorflow.python.tpu.feature_column import _record_variable_scope_and_name
from tensorflow.python.tpu.feature_column import _SUPPORTED_CATEGORICAL_COLUMNS_V2
from tensorflow.python.tpu.feature_column import _SUPPORTED_SEQUENCE_COLUMNS
from tensorflow.python.tpu.feature_column import _TPUBaseEmbeddingColumn
from tensorflow.python.util.tf_export import tf_export
class _TPUSharedDeviceSpecificEmbeddingColumnV2(_TPUSharedEmbeddingColumnV2):
    """TPUSharedEmbeddingColumnV2 which allows serving on TensorCore."""

    def __new__(cls, *args, **kwargs):
        if 'tensor_core_shape' in kwargs:
            cls._tensor_core_shape = kwargs['tensor_core_shape']
            del kwargs['tensor_core_shape']
        if 'embedding_lookup_device' in kwargs:
            cls._embedding_lookup_device = kwargs['embedding_lookup_device']
            del kwargs['embedding_lookup_device']
        return _TPUSharedEmbeddingColumnV2.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        if 'tensor_core_shape' in kwargs:
            self._tensor_core_shape = kwargs['tensor_core_shape']
            del kwargs['tensor_core_shape']
        if 'embedding_lookup_device' in kwargs:
            self._embedding_lookup_device = kwargs['embedding_lookup_device']
            del kwargs['embedding_lookup_device']
        _TPUSharedEmbeddingColumnV2.__init__(self, *args, **kwargs)

    def __deepcopy__(self, memo):
        return _TPUSharedDeviceSpecificEmbeddingColumnV2(*(copy.deepcopy(a, memo) for a in self.__getnewargs__()), tensor_core_shape=self._tensor_core_shape, embedding_lookup_device=self._embedding_lookup_device)

    def _get_dense_tensor_internal(self, transformation_cache, state_manager):
        """Private method that follows _get_dense_tensor_internal."""
        _check_invalid_cases(self._embedding_lookup_device)
        is_cpu = self._embedding_lookup_device == EmbeddingDevice.CPU
        is_cpu = is_cpu or _is_running_on_cpu()
        if is_cpu:
            return super(_TPUSharedDeviceSpecificEmbeddingColumnV2, self)._get_dense_tensor_internal(transformation_cache, state_manager)
        if self._embedding_lookup_device == EmbeddingDevice.TPU_EMBEDDING_CORE:
            return super(_TPUSharedDeviceSpecificEmbeddingColumnV2, self)._get_dense_tensor_internal(transformation_cache, state_manager)
        if tpu.under_tpu_inference_context():
            sparse_tensor = transformation_cache.get(self.categorical_column.name, state_manager)

            def host_computation():
                return pad_sparse_embedding_lookup_indices(sparse_tensor, self._tensor_core_shape[1])
            values, mask = tpu_replication.outside_compilation(host_computation)
        else:
            values = transformation_cache.get(self.categorical_column.name, state_manager)
            mask = transformation_cache.get(self.categorical_column.name + _TENSOR_CORE_MASK_KEY_SUFFIX, state_manager)
        embedding_weights = self.shared_embedding_column_creator.embedding_weights
        return sparse_embedding_aggregate_slice(embedding_weights, (values, mask), self.get_combiner())