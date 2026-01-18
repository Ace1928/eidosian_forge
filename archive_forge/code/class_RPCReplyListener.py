from threading import Event, Lock
from uuid import uuid4
from ncclient.xml_ import *
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport import SessionListener
from ncclient.operations import util
from ncclient.operations.errors import OperationError, TimeoutExpiredError, MissingCapabilityError
import logging
class RPCReplyListener(SessionListener):
    creation_lock = Lock()

    def __new__(cls, session, device_handler):
        with RPCReplyListener.creation_lock:
            instance = session.get_listener_instance(cls)
            if instance is None:
                instance = object.__new__(cls)
                instance._lock = Lock()
                instance._id2rpc = {}
                instance._device_handler = device_handler
                session.add_listener(instance)
                instance.logger = SessionLoggerAdapter(logger, {'session': session})
            return instance

    def register(self, id, rpc):
        with self._lock:
            self._id2rpc[id] = rpc

    def callback(self, root, raw):
        tag, attrs = root
        if self._device_handler.perform_qualify_check():
            if tag != qualify('rpc-reply'):
                return
        if 'message-id' not in attrs:
            raise OperationError("Could not find 'message-id' attribute in <rpc-reply>")
        else:
            id = attrs['message-id']
            with self._lock:
                try:
                    rpc = self._id2rpc[id]
                    self.logger.debug('Delivering to %r', rpc)
                    rpc.deliver_reply(raw)
                except KeyError:
                    raise OperationError("Unknown 'message-id': %s" % id)
                else:
                    del self._id2rpc[id]

    def errback(self, err):
        try:
            for rpc in six.itervalues(self._id2rpc):
                rpc.deliver_error(err)
        finally:
            self._id2rpc.clear()