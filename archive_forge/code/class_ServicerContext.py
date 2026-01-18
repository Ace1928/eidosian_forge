import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class ServicerContext(RpcContext, metaclass=abc.ABCMeta):
    """A context object passed to method implementations."""

    @abc.abstractmethod
    def invocation_metadata(self):
        """Accesses the metadata from the invocation-side of the RPC.

        This method blocks until the value is available or is known not to have been
        emitted from the invocation-side of the RPC.

        Returns:
          The metadata object emitted by the invocation-side of the RPC, or None if
            there was no such value.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def initial_metadata(self, initial_metadata):
        """Accepts the service-side initial metadata value of the RPC.

        This method need not be called by method implementations if they have no
        service-side initial metadata to transmit.

        Args:
          initial_metadata: The service-side initial metadata value of the RPC to
            be transmitted to the invocation side of the RPC.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def terminal_metadata(self, terminal_metadata):
        """Accepts the service-side terminal metadata value of the RPC.

        This method need not be called by method implementations if they have no
        service-side terminal metadata to transmit.

        Args:
          terminal_metadata: The service-side terminal metadata value of the RPC to
            be transmitted to the invocation side of the RPC.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def code(self, code):
        """Accepts the service-side code of the RPC.

        This method need not be called by method implementations if they have no
        code to transmit.

        Args:
          code: The code of the RPC to be transmitted to the invocation side of the
            RPC.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def details(self, details):
        """Accepts the service-side details of the RPC.

        This method need not be called by method implementations if they have no
        service-side details to transmit.

        Args:
          details: The service-side details value of the RPC to be transmitted to
            the invocation side of the RPC.
        """
        raise NotImplementedError()