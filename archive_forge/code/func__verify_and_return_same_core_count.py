import collections
import re
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.eager import remote
from tensorflow.python.framework import config as framework_config
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat
@staticmethod
def _verify_and_return_same_core_count(device_dict):
    """Verifies that every device in device_dict has the same # of cores."""
    num_cores_per_host_set = {len(core_ids) for core_ids in device_dict.values()}
    if len(num_cores_per_host_set) != 1:
        raise RuntimeError('TPU cores on each device is not the same. This should never happen. Devices: {}'.format(device_dict))
    return num_cores_per_host_set.pop()