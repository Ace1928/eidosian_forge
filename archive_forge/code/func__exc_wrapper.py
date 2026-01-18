import contextlib
from kazoo import exceptions as k_exc
from kazoo.protocol import paths
from oslo_serialization import jsonutils
from oslo_utils import strutils
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.utils import kazoo_utils as k_utils
from taskflow.utils import misc
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