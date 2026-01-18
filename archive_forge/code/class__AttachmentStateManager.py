import collections
import contextlib
import logging
import socket
import threading
from oslo_config import cfg
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store.i18n import _LE, _LW
class _AttachmentStateManager(metaclass=AttachmentStateManagerMeta):
    """A global manager of a volume's multiple attachments.

    _AttachmentStateManager manages a _AttachmentState object for the current
    glance node. Primarily it creates one on object initialization and returns
    it via get_state().

    _AttachmentStateManager manages concurrency itself. Independent callers do
    not need to consider interactions between multiple _AttachmentStateManager
    calls when designing their own locking.

    """
    state = None
    use_count = 0
    cond = threading.Condition()

    def __init__(self, host):
        """Initialise a new _AttachmentState

        We will block before creating a new state until all operations
        using a previous state have completed.

        :param host: host
        """
        self.host = host
        while self.use_count != 0:
            self.cond.wait()
        if self.state is None:
            LOG.debug('Initialising _AttachmentStateManager')
            self.state = _AttachmentState()

    @contextlib.contextmanager
    def get_state(self):
        """Return the current attachment state.

        _AttachmentStateManager will not permit a new state object to be
        created while any previous state object is still in use.

        :rtype: _AttachmentState
        """
        with self.cond:
            state = self.state
            if state is None:
                LOG.error('Host not initialized')
                raise exceptions.HostNotInitialized(host=self.host)
            self.use_count += 1
        try:
            LOG.debug('Got _AttachmentState')
            yield state
        finally:
            with self.cond:
                self.use_count -= 1
                self.cond.notify_all()