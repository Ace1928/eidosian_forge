import logging
from threading import Thread, Lock, Event
import ncclient.transport
from ncclient.xml_ import *
from ncclient.capabilities import Capabilities
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport.errors import TransportError, SessionError, SessionCloseError
from ncclient.transport.notify import Notification
class NotificationHandler(SessionListener):

    def __init__(self, notification_q):
        self._notification_q = notification_q

    def callback(self, root, raw):
        tag, _ = root
        if tag == qualify('notification', NETCONF_NOTIFICATION_NS):
            self._notification_q.put(Notification(raw))

    def errback(self, _):
        pass