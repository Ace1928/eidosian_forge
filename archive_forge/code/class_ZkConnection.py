import contextlib
from kazoo import exceptions as k_exc
from kazoo.protocol import paths
from oslo_serialization import jsonutils
from oslo_utils import strutils
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.utils import kazoo_utils as k_utils
from taskflow.utils import misc
class ZkConnection(path_based.PathBasedConnection):

    def __init__(self, backend, client, conf):
        super(ZkConnection, self).__init__(backend)
        self._conf = conf
        self._client = client
        with self._exc_wrapper():
            self._client.start()

    @contextlib.contextmanager
    def _exc_wrapper(self):
        """Exception context-manager which wraps kazoo exceptions.

        This is used to capture and wrap any kazoo specific exceptions and
        then group them into corresponding taskflow exceptions (not doing
        that would expose the underlying kazoo exception model).
        """
        try:
            yield
        except self._client.handler.timeout_exception:
            exc.raise_with_cause(exc.StorageFailure, 'Storage backend timeout')
        except k_exc.SessionExpiredError:
            exc.raise_with_cause(exc.StorageFailure, 'Storage backend session has expired')
        except k_exc.NoNodeError:
            exc.raise_with_cause(exc.NotFound, 'Storage backend node not found')
        except k_exc.NodeExistsError:
            exc.raise_with_cause(exc.Duplicate, 'Storage backend duplicate node')
        except (k_exc.KazooException, k_exc.ZookeeperError):
            exc.raise_with_cause(exc.StorageFailure, 'Storage backend internal error')

    def _join_path(self, *parts):
        return paths.join(*parts)

    def _get_item(self, path):
        with self._exc_wrapper():
            data, _ = self._client.get(path)
        return misc.decode_json(data)

    def _set_item(self, path, value, transaction):
        data = misc.binary_encode(jsonutils.dumps(value))
        if not self._client.exists(path):
            transaction.create(path, data)
        else:
            transaction.set_data(path, data)

    def _del_tree(self, path, transaction):
        for child in self._get_children(path):
            self._del_tree(self._join_path(path, child), transaction)
        transaction.delete(path)

    def _get_children(self, path):
        with self._exc_wrapper():
            return self._client.get_children(path)

    def _ensure_path(self, path):
        with self._exc_wrapper():
            self._client.ensure_path(path)

    def _create_link(self, src_path, dest_path, transaction):
        if not self._client.exists(dest_path):
            transaction.create(dest_path)

    @contextlib.contextmanager
    def _transaction(self):
        transaction = self._client.transaction()
        with self._exc_wrapper():
            yield transaction
            k_utils.checked_commit(transaction)

    def validate(self):
        with self._exc_wrapper():
            try:
                if strutils.bool_from_string(self._conf.get('check_compatible'), default=True):
                    k_utils.check_compatible(self._client, MIN_ZK_VERSION)
            except exc.IncompatibleVersion:
                exc.raise_with_cause(exc.StorageFailure, 'Backend storage is not a compatible version')