import contextlib
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.grappler import _pywrap_tf_cluster as tf_cluster
def DeterminePeakMemoryUsage(self, item):
    """Returns a snapshot of the peak memory usage.

    Args:
      item: The item for which to measure the costs.
    Returns: A hashtable indexed by device name.
    """
    return tf_cluster.TF_DeterminePeakMemoryUsage(item.tf_item, self._tf_cluster)