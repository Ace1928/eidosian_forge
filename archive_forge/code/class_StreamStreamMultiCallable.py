import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class StreamStreamMultiCallable(abc.ABC):
    """Affords invoking a stream-stream RPC in any call style."""

    @abc.abstractmethod
    def __call__(self, request_iterator, timeout, metadata=None, protocol_options=None):
        """Invokes the underlying RPC.

        Args:
          request_iterator: An iterator that yields request values for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of
            the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          An object that is both a Call for the RPC and an iterator of response
            values. Drawing response values from the returned iterator may raise
            AbortionError indicating abortion of the RPC.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def event(self, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        """Asynchronously invokes the underlying RPC.

        Args:
          receiver: A ResponseReceiver to be passed the response data of the RPC.
          abortion_callback: A callback to be called and passed an Abortion value
            in the event of RPC abortion.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of
            the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          A single object that is both a Call object for the RPC and a
            stream.Consumer to which the request values of the RPC should be passed.
        """
        raise NotImplementedError()