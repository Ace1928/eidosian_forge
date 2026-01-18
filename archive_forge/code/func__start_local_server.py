import collections
import re
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.eager import remote
from tensorflow.python.framework import config as framework_config
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat
def _start_local_server(self):
    address = compat.as_text(self._cloud_tpu_client.get_local_ip())
    self._server = server_lib.Server({'local': ['0.0.0.0:0']}, protocol='grpc', config=None, start=True)
    target = compat.as_bytes(self._server.target)
    splits = target.split(compat.as_bytes(':'))
    assert len(splits) == 3, self._server.target
    assert splits[0] == compat.as_bytes('grpc'), self._server.target
    self._coordinator_port = compat.as_text(splits[2])
    self._coordinator_address = '%s:%s' % (address, compat.as_text(self._coordinator_port))