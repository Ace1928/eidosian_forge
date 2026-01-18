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
class _DataServiceDatasetV2(dataset_ops.DatasetSource):
    """A `Dataset` that reads elements from the tf.data service."""

    def __init__(self, dataset_id, processing_mode, address, element_spec, protocol, data_transfer_protocol, job_name=None, consumer_index=None, num_consumers=None, max_outstanding_requests=None, task_refresh_interval_hint_ms=None, cross_trainer_cache=None, target_workers='AUTO'):
        """Constructs a _DataServiceDatasetV2.

    Args:
      dataset_id: The dataset id for the dataset to read from.
      processing_mode: A `tf.data.experimental.service.ShardingPolicy`
        specifying how to shard the dataset among tf.data workers. See
        `tf.data.experimental.service.ShardingPolicy` for details. For backwards
        compatibility, `processing_mode` may also be set to the strings
        `"parallel_epochs"` or `"distributed_epoch"`, which are respectively
        equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.
      address: The tf.data service address, e.g. "localhost:5000".
      element_spec: The dataset element spec for the dataset to read from.
      protocol: The protocol to use for communicating with the tf.data service,
        e.g. "grpc".
      data_transfer_protocol: (Optional.) The protocol to use for transferring
        data with the tf.data service. By default, data is transferred using
        gRPC.
      job_name: (Optional.) The name of the job. If provided, it must be a
        non-empty string or Tensor. This argument makes it possible for multiple
        datasets to share the same job. The default behavior is that the dataset
        creates anonymous, exclusively owned jobs.
      consumer_index: (Optional.) The index of the consumer in the range from
        `0` to `num_consumers`. Must be specified alongside `num_consumers`.
        When specified, consumers will read from the job in a strict round-robin
        order, instead of the default first-come-first-served order.
      num_consumers: (Optional.) The number of consumers which will consume from
        the job. Must be specified alongside `consumer_index`. When specified,
        consumers will read from the job in a strict round-robin order, instead
        of the default first-come-first-served order. When `num_consumers` is
        specified, the dataset must have infinite cardinality to prevent a
        producer from running out of data early and causing consumers to go out
        of sync.
      max_outstanding_requests: (Optional.) A limit on how many elements may be
        requested at the same time. You can use this option to control the
        amount of memory used, since `distribute` won't use more than
        `element_size` * `max_outstanding_requests` of memory.
      task_refresh_interval_hint_ms: (Optional.) A hint for how often to query
        the dispatcher for task changes.
      cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is
        provided, dataset iteration will be shared across concurrently running
        trainers. See
        https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers
        for details.
      target_workers: (Optional.) Which workers to read from. If `"AUTO"`,
        tf.data runtime decides which workers to read from. If `"ANY"`, reads
        from any tf.data service workers. If `"LOCAL"`, only reads from local
        in-processs tf.data service workers. `"AUTO"` works well for most cases,
        while users can specify other targets. For example, `"LOCAL"` helps
        avoid RPCs and data copy if every TF worker colocates with a tf.data
        service worker. Consumers of a shared job must use the same
        `target_workers`. Defaults to `"AUTO"`.
    """
        if consumer_index is None != num_consumers is None:
            raise ValueError('Must either set both `consumer_index` and `num_consumers`, or neither. ', f'consumer_index={consumer_index}, num_consumers={num_consumers}')
        if num_consumers is not None and job_name is None:
            raise ValueError(f'`job_name` must be set when setting `num_consumers`. num_consumers was set to {num_consumers}.')
        processing_mode_def = data_service_pb2.ProcessingModeDef(sharding_policy=_get_validated_sharding_policy(processing_mode)._to_proto())
        if job_name is None:
            job_name = ''
        if max_outstanding_requests is None:
            max_outstanding_requests = dataset_ops.AUTOTUNE
        if task_refresh_interval_hint_ms is None:
            task_refresh_interval_hint_ms = dataset_ops.AUTOTUNE
        self._dataset_id = _to_tensor(dataset_id)
        self._processing_mode = ops.convert_to_tensor(processing_mode_def.SerializeToString(), dtype=dtypes.string, name='processing_mode')
        self._address = ops.convert_to_tensor(address, dtype=dtypes.string, name='address')
        self._protocol = ops.convert_to_tensor(protocol, dtype=dtypes.string, name='protocol')
        self._job_name = ops.convert_to_tensor(job_name, dtype=dtypes.string, name='job_name')
        self._consumer_index = ops.convert_to_tensor(-1 if consumer_index is None else consumer_index, dtype=dtypes.int64, name='consumer_index')
        self._num_consumers = ops.convert_to_tensor(-1 if num_consumers is None else num_consumers, dtype=dtypes.int64, name='num_consumers')
        self._max_outstanding_requests = ops.convert_to_tensor(max_outstanding_requests, dtype=dtypes.int64, name='max_outstanding_requests')
        self._element_spec = element_spec
        uncompress_func = structured_function.StructuredFunctionWrapper(lambda x: compression_ops.uncompress(x, output_spec=element_spec), transformation_name='DataServiceDataset.uncompress()', input_structure=tensor.TensorSpec(shape=(), dtype=dtypes.variant))
        cross_trainer_cache_options = cross_trainer_cache._to_proto().SerializeToString() if cross_trainer_cache else None
        compat_kwargs = {}
        if data_transfer_protocol is not None:
            compat_kwargs['data_transfer_protocol'] = data_transfer_protocol
        uncompress = True
        variant_tensor = gen_experimental_dataset_ops.data_service_dataset_v4(dataset_id=self._dataset_id, processing_mode=self._processing_mode, address=self._address, protocol=self._protocol, job_name=self._job_name, consumer_index=self._consumer_index, num_consumers=self._num_consumers, max_outstanding_requests=self._max_outstanding_requests, task_refresh_interval_hint_ms=task_refresh_interval_hint_ms, iteration_counter=gen_experimental_dataset_ops.dummy_iteration_counter(), target_workers=target_workers, uncompress=uncompress, uncompress_fn=uncompress_func.function, cross_trainer_cache_options=cross_trainer_cache_options, **compat_kwargs, **self._flat_structure)
        super(_DataServiceDatasetV2, self).__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec