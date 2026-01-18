import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class ResponseReceiver(abc.ABC):
    """Invocation-side object used to accept the output of an RPC."""

    @abc.abstractmethod
    def initial_metadata(self, initial_metadata):
        """Receives the initial metadata from the service-side of the RPC.

        Args:
          initial_metadata: The initial metadata object emitted from the
            service-side of the RPC.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def response(self, response):
        """Receives a response from the service-side of the RPC.

        Args:
          response: A response object emitted from the service-side of the RPC.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def complete(self, terminal_metadata, code, details):
        """Receives the completion values emitted from the service-side of the RPC.

        Args:
          terminal_metadata: The terminal metadata object emitted from the
            service-side of the RPC.
          code: The code object emitted from the service-side of the RPC.
          details: The details object emitted from the service-side of the RPC.
        """
        raise NotImplementedError()