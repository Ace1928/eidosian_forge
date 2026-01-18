import contextlib
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.grappler import _pywrap_tf_cluster as tf_cluster
def ListDevices(self):
    """Returns a list of available hardware devices."""
    if self._tf_cluster is None:
        return []
    return [device_properties_pb2.NamedDevice.FromString(device) for device in tf_cluster.TF_ListDevices(self._tf_cluster)]