import numpy as np
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
def _tpu_host_device_name(job, task):
    """Returns the device name for the CPU device on `task` of `job`."""
    if job is None:
        return '/task:%d/device:CPU:0' % task
    else:
        return '/job:%s/task:%d/device:CPU:0' % (job, task)