import contextlib
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.grappler import _pywrap_tf_cluster as tf_cluster
def EstimatePerformance(self, device):
    return tf_cluster.TF_EstimatePerformance(device.SerializeToString())