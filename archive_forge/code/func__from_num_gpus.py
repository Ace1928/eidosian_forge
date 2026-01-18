from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.util.tf_export import tf_export
@classmethod
def _from_num_gpus(cls, num_gpus):
    return cls(device_util.local_devices_from_num_gpus(num_gpus))