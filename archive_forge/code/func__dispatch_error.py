import logging
from threading import Thread, Lock, Event
import ncclient.transport
from ncclient.xml_ import *
from ncclient.capabilities import Capabilities
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport.errors import TransportError, SessionError, SessionCloseError
from ncclient.transport.notify import Notification
def _dispatch_error(self, err):
    with self._lock:
        listeners = list(self._listeners)
    for l in listeners:
        self.logger.debug('dispatching error to %r', l)
        try:
            l.errback(err)
        except Exception as e:
            self.logger.warning('error dispatching to %r: %r', l, e)