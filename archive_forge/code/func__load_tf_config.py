import json
import os
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def _load_tf_config(port):
    assert all([x in os.environ for x in [_SM_CURRENT_HOST, _SM_HOSTS]]), 'Not a SageMaker Environment'
    hosts = sorted(json.loads(os.environ[_SM_HOSTS])) if os.environ[_SM_HOSTS] != '' else []
    current_host = os.environ[_SM_CURRENT_HOST]
    if current_host not in hosts:
        return {}
    host_index = hosts.index(current_host)
    hosts = ['%s:%s' % (host, port) for host in hosts]
    tf_config = {_CLUSTER_KEY: {_WORKER_KEY: hosts}, _TASK_KEY: {_TYPE_KEY: _WORKER_KEY, _INDEX_KEY: host_index}}
    return tf_config