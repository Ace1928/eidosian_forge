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
def _get_device_dict_and_cores(devices):
    """Returns a dict of hosts to cores and total cores given devices names.

    Returns a namedtuple with two attributes:
      device_map: A map of host_ids to a list of core_ids.
      total_cores: The total number of cores within the TPU system.

    Args:
      devices: A list of devices returned by session.list_devices()
    """
    device_map = collections.defaultdict(list)
    num_cores = 0
    for device in devices:
        match = _TPU_DEVICE_REGEX.match(device.name)
        if match:
            host_id = match.group('host_id')
            core_id = match.group('core_id')
            device_map[host_id].append(core_id)
            num_cores += 1
    return DeviceDetails(device_map, num_cores)