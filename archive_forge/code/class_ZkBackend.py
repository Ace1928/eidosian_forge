import contextlib
from kazoo import exceptions as k_exc
from kazoo.protocol import paths
from oslo_serialization import jsonutils
from oslo_utils import strutils
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.utils import kazoo_utils as k_utils
from taskflow.utils import misc
class ZkBackend(path_based.PathBasedBackend):
    """A zookeeper-backed backend.

    Example configuration::

        conf = {
            "hosts": "192.168.0.1:2181,192.168.0.2:2181,192.168.0.3:2181",
            "path": "/taskflow",
        }

    Do note that the creation of a kazoo client is achieved
    by :py:func:`~taskflow.utils.kazoo_utils.make_client` and the transfer
    of this backend configuration to that function to make a
    client may happen at ``__init__`` time. This implies that certain
    parameters from this backend configuration may be provided to
    :py:func:`~taskflow.utils.kazoo_utils.make_client` such
    that if a client was not provided by the caller one will be created
    according to :py:func:`~taskflow.utils.kazoo_utils.make_client`'s
    specification
    """
    DEFAULT_PATH = '/taskflow'

    def __init__(self, conf, client=None):
        super(ZkBackend, self).__init__(conf)
        if not paths.isabs(self._path):
            raise ValueError('Zookeeper path must be absolute')
        if client is not None:
            self._client = client
            self._owned = False
        else:
            self._client = k_utils.make_client(self._conf)
            self._owned = True
        self._validated = False

    def get_connection(self):
        conn = ZkConnection(self, self._client, self._conf)
        if not self._validated:
            conn.validate()
            self._validated = True
        return conn

    def close(self):
        self._validated = False
        if not self._owned:
            return
        try:
            k_utils.finalize_client(self._client)
        except (k_exc.KazooException, k_exc.ZookeeperError):
            exc.raise_with_cause(exc.StorageFailure, 'Unable to finalize client')