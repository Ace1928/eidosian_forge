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
class _DataServiceDatasetV1(dataset_ops.DatasetV1Adapter):
    """A `Dataset` that executes its input through the tf.data service."""

    @functools.wraps(_DataServiceDatasetV2.__init__)
    def __init__(self, dataset_id, processing_mode, address, element_spec, protocol, data_transfer_protocol, job_name, consumer_index, num_consumers, max_outstanding_requests, task_refresh_interval_hint_ms, cross_trainer_cache, target_workers):
        self._wrapped = _DataServiceDatasetV2(dataset_id=dataset_id, processing_mode=processing_mode, address=address, element_spec=element_spec, protocol=protocol, data_transfer_protocol=data_transfer_protocol, job_name=job_name, consumer_index=consumer_index, num_consumers=num_consumers, max_outstanding_requests=max_outstanding_requests, task_refresh_interval_hint_ms=task_refresh_interval_hint_ms, cross_trainer_cache=cross_trainer_cache, target_workers=target_workers)
        super(_DataServiceDatasetV1, self).__init__(self._wrapped)