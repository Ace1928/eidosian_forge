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
def _register_dataset(service, dataset, compression, dataset_id=None):
    """Registers a dataset with the tf.data service.

  This transformation is similar to `register_dataset`, but supports additional
  parameters which we do not yet want to add to the public Python API.

  Args:
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
        address and `<protocol>` can optionally be used to override the default
        protocol to use. If it's a tuple, it should be (protocol, address).
    dataset: A `tf.data.Dataset` to register with the tf.data service.
    compression: How to compress the dataset's elements before transferring them
      over the network. "AUTO" leaves the decision of how to compress up to the
      tf.data service runtime. `None` indicates not to compress.
    dataset_id: (Optional.) By default, tf.data service generates a unique
      (string) ID for each registered dataset. If a `dataset_id` is provided, it
      will use the specified ID. If a dataset with a matching ID already exists,
      no new dataset is registered. This is useful if multiple training jobs
      want to (re)use the same dataset for training. In this case, they can
      register the dataset with the same dataset ID.

  Returns:
    A scalar string tensor representing the dataset ID.
  """
    _validate_compression(compression)
    if isinstance(service, tuple):
        protocol, address = service
    else:
        protocol, address = _parse_service(service)
    external_state_policy = dataset.options().experimental_external_state_policy
    if external_state_policy is None:
        external_state_policy = ExternalStatePolicy.WARN
    encoded_spec = None
    if context.executing_eagerly():
        encoded_spec = nested_structure_coder.encode_structure(dataset.element_spec).SerializeToString()
    if compression == COMPRESSION_AUTO:
        dataset = dataset.map(lambda *x: compression_ops.compress(x), num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset._apply_debug_options()
    metadata = data_service_pb2.DataServiceMetadata(element_spec=encoded_spec, compression=_get_compression_proto(compression))
    return gen_experimental_dataset_ops.register_dataset_v2(dataset._variant_tensor, address=address, protocol=protocol, external_state_policy=external_state_policy.value, requested_dataset_id=dataset_id, metadata=metadata.SerializeToString())