from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
class GraphTracingReachedDestination(Exception):
    pass