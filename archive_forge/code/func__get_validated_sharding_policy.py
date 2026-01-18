import enum
import functools
from tensorflow.core.protobuf import data_service_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import compression_ops
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def _get_validated_sharding_policy(processing_mode):
    """Validates `processing_mode` and converts it to ShardingPolicy."""
    if isinstance(processing_mode, ShardingPolicy):
        return processing_mode
    if processing_mode == _PARALLEL_EPOCHS:
        return ShardingPolicy.OFF
    if processing_mode == _DISTRIBUTED_EPOCH:
        return ShardingPolicy.DYNAMIC
    raise ValueError(f'tf.data service processing mode should be a `tf.data.experimental.service.ShardingPolicy`, `"parallel_epochs"`, or `"distributed_epoch"`. Got {processing_mode!r}.')