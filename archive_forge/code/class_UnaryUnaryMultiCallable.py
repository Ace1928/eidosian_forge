import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class UnaryUnaryMultiCallable(abc.ABC):
    """Affords invoking a unary-unary RPC in any call style."""

    @abc.abstractmethod
    def __call__(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        """Synchronously invokes the underlying RPC.

        Args:
          request: The request value for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of
            the RPC.
          with_call: Whether or not to include return a Call for the RPC in addition
            to the response.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          The response value for the RPC, and a Call for the RPC if with_call was
            set to True at invocation.

        Raises:
          AbortionError: Indicating that the RPC was aborted.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def future(self, request, timeout, metadata=None, protocol_options=None):
        """Asynchronously invokes the underlying RPC.

        Args:
          request: The request value for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of
            the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          An object that is both a Call for the RPC and a future.Future. In the
            event of RPC completion, the return Future's result value will be the
            response value of the RPC. In the event of RPC abortion, the returned
            Future's exception value will be an AbortionError.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def event(self, request, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        """Asynchronously invokes the underlying RPC.

        Args:
          request: The request value for the RPC.
          receiver: A ResponseReceiver to be passed the response data of the RPC.
          abortion_callback: A callback to be called and passed an Abortion value
            in the event of RPC abortion.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of
            the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          A Call for the RPC.
        """
        raise NotImplementedError()