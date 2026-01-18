import abc
import threading
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_messaging import exceptions
class IncomingMessage(object, metaclass=abc.ABCMeta):
    """The IncomingMessage class represents a single message received from the
    messaging backend. Instances of this class are passed to up a server's
    messaging processing logic. The backend driver must provide a concrete
    derivation of this class which provides the backend specific logic for its
    public methods.

    :param ctxt: Context metadata provided by sending application.
    :type ctxt: dict
    :param message: The message as provided by the sending application.
    :type message: dict
    """

    def __init__(self, ctxt, message):
        self.ctxt = ctxt
        self.message = message
        self.client_timeout = None

    def acknowledge(self):
        """Called by the server to acknowledge receipt of the message. When
        this is called the driver must notify the backend of the
        acknowledgment. This call should block at least until the driver has
        processed the acknowledgment request locally. It may unblock before the
        acknowledgment state has been acted upon by the backend.

        If the acknowledge operation fails this method must issue a log message
        describing the reason for the failure.

        :raises: Does not raise an exception
        """

    @abc.abstractmethod
    def requeue(self):
        """Called by the server to return the message to the backend so it may
        be made available for consumption by another server.  This call should
        block at least until the driver has processed the requeue request
        locally. It may unblock before the backend makes the requeued message
        available for consumption.

        If the requeue operation fails this method must issue a log message
        describing the reason for the failure.

        Support for this method is _optional_.  The
        :py:meth:`BaseDriver.require_features` method should indicate whether
        or not support for requeue is available.

        :raises: Does not raise an exception
        """