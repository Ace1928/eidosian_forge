import threading
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.grappler import _pywrap_tf_optimizer as tf_opt
from tensorflow.python.grappler import cluster as gcluster
def OptimizeGraph(config_proto, metagraph, verbose=True, graph_id=b'graph_to_optimize', cluster=None, strip_default_attributes=False):
    """Optimize the provided metagraph.

  For best results, the signature_def field in `metagraph` should be populated
  with information about input (feed) and output (fetch) tensors.

  Args:
    config_proto: a ConfigProto protobuf.
    metagraph: a MetagraphDef protobuf.
    verbose: whether to log optimization results.
    graph_id: a string identifying this graph.
    cluster: a grappler cluster object representing hardware resources
        available to run this graph.
    strip_default_attributes: whether graph node attributes having default
        values should be removed after all the optimization passes. This
        option is useful if the resulting graph will be executed by an older
        process that might not know some of the recently added attributes.
  """
    if not isinstance(config_proto, config_pb2.ConfigProto):
        raise TypeError(f'Argument `config_proto` should be a tf.ConfigProto, received type: {type(config_proto).__name__}')
    if cluster is not None:
        out_graph = tf_opt.TF_OptimizeGraph(cluster.tf_cluster, config_proto.SerializeToString(), metagraph.SerializeToString(), verbose, graph_id, strip_default_attributes)
    else:
        with _OPTIMIZE_GRAPH_CLUSTER_LOCK:
            cluster = gcluster.Cluster()
            try:
                out_graph = tf_opt.TF_OptimizeGraph(cluster.tf_cluster, config_proto.SerializeToString(), metagraph.SerializeToString(), verbose, graph_id, strip_default_attributes)
            finally:
                cluster.Shutdown()
    return graph_pb2.GraphDef().FromString(out_graph)