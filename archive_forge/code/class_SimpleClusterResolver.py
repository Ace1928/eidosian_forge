import abc
import collections
import six
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.cluster_resolver.SimpleClusterResolver')
class SimpleClusterResolver(ClusterResolver):
    """Simple implementation of ClusterResolver that accepts all attributes.

  Please see the base class for documentation of arguments of its constructor.

  It is useful if you want to specify some or all attributes.

  Usage example with `tf.distribute.Strategy`:

    ```Python
    cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                               "worker1.example.com:2222"]})

    # On worker 0
    cluster_resolver = SimpleClusterResolver(cluster, task_type="worker",
                                             task_id=0,
                                             num_accelerators={"GPU": 8},
                                             rpc_layer="grpc")
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)

    # On worker 1
    cluster_resolver = SimpleClusterResolver(cluster, task_type="worker",
                                             task_id=1,
                                             num_accelerators={"GPU": 8},
                                             rpc_layer="grpc")
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)
    ```
  """

    def __init__(self, cluster_spec, master='', task_type=None, task_id=None, environment='', num_accelerators=None, rpc_layer=None):
        """Creates a SimpleClusterResolver from a ClusterSpec."""
        super(SimpleClusterResolver, self).__init__()
        self._task_type = task_type
        self._task_id = task_id
        self._environment = environment
        self._num_accelerators = num_accelerators
        self._rpc_layer = rpc_layer
        if not isinstance(cluster_spec, ClusterSpec):
            raise TypeError('cluster_spec must be a `tf.train.ClusterSpec`.')
        self._cluster_spec = cluster_spec
        if not isinstance(master, str):
            raise TypeError('master must be a string.')
        self._master = master

    def cluster_spec(self):
        """Returns the ClusterSpec passed into the constructor."""
        return self._cluster_spec

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        """Returns the master address to use when creating a session.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC used by distributed TensorFlow.

    Returns:
      The name or URL of the session master.

    If a task_type and task_id is given, this will override the `master`
    string passed into the initialization function.
    """
        if task_type is not None and task_id is not None:
            master = self.cluster_spec().task_address(task_type, task_id)
        else:
            master = self._master
        return format_master_url(master, rpc_layer=rpc_layer or self._rpc_layer)

    @property
    def task_type(self):
        return self._task_type

    @property
    def task_id(self):
        return self._task_id

    @task_type.setter
    def task_type(self, task_type):
        self._task_type = task_type

    @task_id.setter
    def task_id(self, task_id):
        self._task_id = task_id

    @property
    def environment(self):
        return self._environment

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        """Returns the number of accelerator cores per worker.

    The SimpleClusterResolver does not do automatic detection of accelerators,
    and thus all arguments are unused and we simply return the value provided
    in the constructor.

    Args:
      task_type: Unused.
      task_id: Unused.
      config_proto: Unused.
    """
        del task_type, task_id, config_proto
        if self._num_accelerators is None:
            return {}
        return self._num_accelerators

    @property
    def rpc_layer(self):
        return self._rpc_layer

    @rpc_layer.setter
    def rpc_layer(self, rpc_layer):
        self._rpc_layer = rpc_layer