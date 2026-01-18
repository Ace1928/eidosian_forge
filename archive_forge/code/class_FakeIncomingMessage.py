import copy
import queue
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import base
class FakeIncomingMessage(base.RpcIncomingMessage):

    def __init__(self, ctxt, message, reply_q, requeue):
        super(FakeIncomingMessage, self).__init__(ctxt, message)
        self.requeue_callback = requeue
        self._reply_q = reply_q

    def reply(self, reply=None, failure=None):
        if self._reply_q:
            failure = failure[1] if failure else None
            self._reply_q.put((reply, failure))

    def requeue(self):
        self.requeue_callback()

    def heartbeat(self):
        """Heartbeat is not supported."""