import collections
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu
from tensorflow.python.util.tf_export import tf_export
@tf_export('tpu.experimental.TPUSystemMetadata')
class TPUSystemMetadata(collections.namedtuple('TPUSystemMetadata', ['num_cores', 'num_hosts', 'num_of_cores_per_host', 'topology', 'devices'])):
    """Describes some metadata about the TPU system.

  Attributes:
    num_cores: interger. Total number of TPU cores in the TPU system.
    num_hosts: interger. Total number of hosts (TPU workers) in the TPU system.
    num_of_cores_per_host: interger. Number of TPU cores per host (TPU worker).
    topology: an instance of `tf.tpu.experimental.Topology`, which describes the
      physical topology of TPU system.
    devices: a tuple of strings, which describes all the TPU devices in the
      system.
  """

    def __new__(cls, num_cores, num_hosts, num_of_cores_per_host, topology, devices):
        return super(TPUSystemMetadata, cls).__new__(cls, num_cores, num_hosts, num_of_cores_per_host, topology, devices)