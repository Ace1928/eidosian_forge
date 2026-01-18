import logging
from threading import Thread, Lock, Event
import ncclient.transport
from ncclient.xml_ import *
from ncclient.capabilities import Capabilities
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport.errors import TransportError, SessionError, SessionCloseError
from ncclient.transport.notify import Notification
def _post_connect(self, timeout=60):
    """Greeting stuff"""
    init_event = Event()
    error = [None]

    def ok_cb(id, capabilities):
        self._id = id
        self._server_capabilities = capabilities
        init_event.set()

    def err_cb(err):
        error[0] = err
        init_event.set()
    self.add_listener(NotificationHandler(self._notification_q))
    listener = HelloHandler(ok_cb, err_cb)
    self.add_listener(listener)
    self.send(HelloHandler.build(self._client_capabilities, self._device_handler))
    self.logger.debug('starting main loop')
    self.start()
    init_event.wait(timeout)
    if not init_event.is_set():
        raise SessionError('Capability exchange timed out')
    self.remove_listener(listener)
    if error[0]:
        raise error[0]
    if 'urn:ietf:params:netconf:base:1.1' in self._server_capabilities and 'urn:ietf:params:netconf:base:1.1' in self._client_capabilities:
        self.logger.debug("After 'hello' message selecting netconf:base:1.1 for encoding")
        self._base = NetconfBase.BASE_11
    self.logger.info('initialized: session-id=%s | server_capabilities=%s', self._id, self._server_capabilities)