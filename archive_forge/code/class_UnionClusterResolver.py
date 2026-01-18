import abc
import collections
import six
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.cluster_resolver.UnionResolver')
class UnionClusterResolver(ClusterResolver):
    """Performs a union on underlying ClusterResolvers.

  This class performs a union given two or more existing ClusterResolvers. It
  merges the underlying ClusterResolvers, and returns one unified ClusterSpec
  when cluster_spec is called. The details of the merge function is
  documented in the cluster_spec function.

  For additional ClusterResolver properties such as task type, task index,
  rpc layer, environment, etc..., we will return the value from the first
  ClusterResolver in the union.

  An example to combine two cluster resolvers:

    ```Python
    cluster_0 = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                                 "worker1.example.com:2222"]})
    cluster_resolver_0 = SimpleClusterResolver(cluster, task_type="worker",
                                               task_id=0,
                                               rpc_layer="grpc")

    cluster_1 = tf.train.ClusterSpec({"ps": ["ps0.example.com:2222",
                                             "ps1.example.com:2222"]})
    cluster_resolver_1 = SimpleClusterResolver(cluster, task_type="ps",
                                               task_id=0,
                                               rpc_layer="grpc")

    # Its task type would be "worker".
    cluster_resolver = UnionClusterResolver(cluster_resolver_0,
                                            cluster_resolver_1)
    ```

  An example to override the number of GPUs in a TFConfigClusterResolver
  instance:

    ```Python
    tf_config = TFConfigClusterResolver()
    gpu_override = SimpleClusterResolver(tf_config.cluster_spec(),
                                         num_accelerators={"GPU": 1})
    cluster_resolver = UnionResolver(gpu_override, tf_config)
    ```
  """

    def __init__(self, *args, **kwargs):
        """Initializes a UnionClusterResolver with other ClusterResolvers.

    Args:
      *args: `ClusterResolver` objects to be unionized.
      **kwargs:
        rpc_layer - (Optional) Override value for the RPC layer used by
          TensorFlow.
        task_type - (Optional) Override value for the current task type.
        task_id - (Optional) Override value for the current task index.

    Raises:
      TypeError: If any argument is not a subclass of `ClusterResolvers`.
      ValueError: If there are no arguments passed.
    """
        super(UnionClusterResolver, self).__init__()
        self._rpc_layer = kwargs.pop('rpc_layer', None)
        self._task_type = kwargs.pop('task_type', None)
        self._task_id = kwargs.pop('task_id', None)
        if kwargs:
            raise ValueError('Unexpected kwargs provided {!r}'.format(kwargs))
        if not args:
            raise ValueError('At least one ClusterResolver is required.')
        for cluster_resolver in args:
            if not isinstance(cluster_resolver, ClusterResolver):
                raise TypeError('All arguments must be a sub-class of `ClusterResolver.`')
        self._cluster_resolvers = args

    def cluster_spec(self):
        """Returns a union of all the ClusterSpecs from the ClusterResolvers.

    Returns:
      A ClusterSpec containing host information merged from all the underlying
      ClusterResolvers.

    Raises:
      KeyError: If there are conflicting keys detected when merging two or
      more dictionaries, this exception is raised.

    Note: If there are multiple ClusterResolvers exposing ClusterSpecs with the
    same job name, we will merge the list/dict of workers.

    If *all* underlying ClusterSpecs expose the set of workers as lists, we will
    concatenate the lists of workers, starting with the list of workers from
    the first ClusterResolver passed into the constructor.

    If *any* of the ClusterSpecs expose the set of workers as a dict, we will
    treat all the sets of workers as dicts (even if they are returned as lists)
    and will only merge them into a dict if there is no conflicting keys. If
    there is a conflicting key, we will raise a `KeyError`.
    """
        merged_cluster = {}
        for cluster_resolver in self._cluster_resolvers:
            cluster_spec = cluster_resolver.cluster_spec()
            cluster_dict = cluster_spec.as_dict()
            for job_name, tasks in cluster_dict.items():
                if job_name in merged_cluster:
                    if isinstance(tasks, dict):
                        merged_cluster[job_name] = {}
                elif isinstance(tasks, list):
                    merged_cluster[job_name] = []
                else:
                    merged_cluster[job_name] = {}
        for cluster_resolver in self._cluster_resolvers:
            cluster_spec = cluster_resolver.cluster_spec()
            cluster_dict = cluster_spec.as_dict()
            for job_name, tasks in cluster_dict.items():
                if isinstance(merged_cluster[job_name], list):
                    merged_cluster[job_name].extend(tasks)
                else:
                    if isinstance(tasks, list):
                        task_dict = dict(zip(range(0, len(tasks)), tasks))
                    else:
                        task_dict = tasks.copy()
                    task_keys = set(task_dict)
                    merged_keys = set(merged_cluster[job_name].keys())
                    intersected_keys = task_keys.intersection(merged_keys)
                    if intersected_keys:
                        raise KeyError('Duplicate keys detected when merging two ClusterSpecs: %s' % repr(intersected_keys))
                    merged_cluster[job_name].update(task_dict)
        return ClusterSpec(merged_cluster)

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        """Returns the master address to use when creating a session.

    This usually returns the master from the first ClusterResolver passed in,
    but you can override this by specifying the task_type and task_id.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.
    """
        if task_type is not None and task_id is not None:
            master = self.cluster_spec().task_address(task_type, task_id)
            return format_master_url(master, rpc_layer or self._rpc_layer)
        return self._cluster_resolvers[0].master(rpc_layer=rpc_layer)

    @property
    def task_type(self):
        return self._task_type or self._cluster_resolvers[0].task_type

    @property
    def task_id(self):
        return self._task_id or self._cluster_resolvers[0].task_id

    @task_type.setter
    def task_type(self, task_type):
        self._task_type = task_type

    @task_id.setter
    def task_id(self, task_id):
        self._task_id = task_id

    @property
    def environment(self):
        return self._cluster_resolvers[0].environment

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        return self._cluster_resolvers[0].num_accelerators(task_type, task_id, config_proto)

    @property
    def rpc_layer(self):
        return self._rpc_layer or self._cluster_resolvers[0].rpc_layer

    @rpc_layer.setter
    def rpc_layer(self, rpc_layer):
        self._rpc_layer = rpc_layer