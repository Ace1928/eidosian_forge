import logging
from threading import Thread, Lock, Event
import ncclient.transport
from ncclient.xml_ import *
from ncclient.capabilities import Capabilities
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport.errors import TransportError, SessionError, SessionCloseError
from ncclient.transport.notify import Notification
class SessionListener(object):
    """Base class for :class:`Session` listeners, which are notified when a new
    NETCONF message is received or an error occurs.

    .. note::
        Avoid time-intensive tasks in a callback's context.
    """

    def callback(self, root, raw):
        """Called when a new XML document is received. The *root* argument allows the callback to determine whether it wants to further process the document.

        Here, *root* is a tuple of *(tag, attributes)* where *tag* is the qualified name of the root element and *attributes* is a dictionary of its attributes (also qualified names).

        *raw* will contain the XML document as a string.
        """
        raise NotImplementedError

    def errback(self, ex):
        """Called when an error occurs.

        :type ex: :exc:`Exception`
        """
        raise NotImplementedError