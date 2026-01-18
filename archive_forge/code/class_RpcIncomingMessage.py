import abc
import threading
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_messaging import exceptions
class RpcIncomingMessage(IncomingMessage, metaclass=abc.ABCMeta):
    """The RpcIncomingMessage represents an RPC request message received from
    the backend. This class must be used for RPC calls that return a value to
    the caller.
    """

    @abc.abstractmethod
    def reply(self, reply=None, failure=None):
        """Called by the server to send an RPC reply message or an exception
        back to the calling client.

        If an exception is passed via *failure* the driver must convert it to
        a form that can be sent as a message and properly converted back to the
        exception at the remote.

        The driver must provide a way to determine the destination address for
        the reply. For example the driver may use the *reply-to* field from the
        corresponding incoming message. Often a driver will also need to set a
        correlation identifier in the reply to help the remote route the reply
        to the correct RPCClient.

        The driver should provide an *at-most-once* delivery guarantee for
        reply messages. This call should block at least until the reply message
        has been handed off to the backend - there is no need to confirm that
        the reply has been delivered.

        If the reply operation fails this method must issue a log message
        describing the reason for the failure.

        See :py:meth:`BaseDriver.send` for details regarding how the received
        reply is processed.

        :param reply: reply message body
        :type reply: dict
        :param failure: an exception thrown by the RPC call
        :type failure: Exception
        :raises: Does not raise an exception
        """

    @abc.abstractmethod
    def heartbeat(self):
        """Called by the server to send an RPC heartbeat message back to
        the calling client.

        If the client (is new enough to have) passed its timeout value during
        the RPC call, this method will be called periodically by the server
        to update the client's timeout timer while a long-running call is
        executing.

        :raises: Does not raise an exception
        """