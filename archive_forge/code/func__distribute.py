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
def _distribute(processing_mode, service, job_name=None, consumer_index=None, num_consumers=None, max_outstanding_requests=None, task_refresh_interval_hint_ms=None, data_transfer_protocol=None, compression='AUTO', cross_trainer_cache=None, target_workers='AUTO'):
    """A transformation that moves dataset processing to the tf.data service.

  This transformation is similar to `distribute`, but supports additional
  parameters which we do not yet want to add to the public Python API.

  Args:
    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying
      how to shard the dataset among tf.data workers. See
      `tf.data.experimental.service.ShardingPolicy` for details. For backwards
      compatibility, `processing_mode` may also be set to the strings
      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively
      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
        address and `<protocol>` can optionally be used to override the default
        protocol to use. If it's a tuple, it should be (protocol, address).
    job_name: (Optional.) The name of the job. If provided, it must be a
      non-empty string. This argument makes it possible for multiple datasets to
      share the same job. The default behavior is that the dataset creates
      anonymous, exclusively owned jobs.
    consumer_index: (Optional.) The index of the consumer in the range from `0`
      to `num_consumers`. Must be specified alongside `num_consumers`. When
      specified, consumers will read from the job in a strict round-robin order,
      instead of the default first-come-first-served order.
    num_consumers: (Optional.) The number of consumers which will consume from
      the job. Must be specified alongside `consumer_index`. When specified,
      consumers will read from the job in a strict round-robin order, instead of
      the default first-come-first-served order. When `num_consumers` is
      specified, the dataset must have infinite cardinality to prevent a
      producer from running out of data early and causing consumers to go out of
      sync.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.
    task_refresh_interval_hint_ms: (Optional.) A hint for how often to query the
      dispatcher for task changes.
    data_transfer_protocol: (Optional.) The protocol to use for transferring
      data with the tf.data service. By default, data is transferred using gRPC.
    compression: How to compress the dataset's elements before transferring them
      over the network. "AUTO" leaves the decision of how to compress up to the
      tf.data service runtime. `None` indicates not to compress.
    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is
      provided, dataset iteration will be shared across concurrently running
      trainers. See
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers
      for details.
    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data
      runtime decides which workers to read from. If `"ANY"`, reads from any
      tf.data service workers. If `"LOCAL"`, only reads from local in-processs
      tf.data service workers. `"AUTO"` works well for most cases, while users
      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and
      data copy if every TF worker colocates with a tf.data service worker.
      Consumers of a shared job must use the same `target_workers`. Defaults to
      `"AUTO"`.

  Returns:
    Dataset: A `Dataset` of the elements produced by the data service.
  """
    processing_mode = _get_validated_sharding_policy(processing_mode)
    _validate_compression(compression)

    def _apply_fn(dataset):
        dataset_id = _register_dataset(service, dataset, compression=compression)
        return _from_dataset_id(processing_mode, service, dataset_id, dataset.element_spec, job_name=job_name, consumer_index=consumer_index, num_consumers=num_consumers, max_outstanding_requests=max_outstanding_requests, task_refresh_interval_hint_ms=task_refresh_interval_hint_ms, data_transfer_protocol=data_transfer_protocol, cross_trainer_cache=cross_trainer_cache, target_workers=target_workers)
    return _apply_fn