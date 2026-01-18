import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
@abc.abstractmethod
def add_abortion_callback(self, abortion_callback):
    """Registers a callback to be called if the RPC is aborted.

        Args:
          abortion_callback: A callable to be called and passed an Abortion value
            in the event of RPC abortion.
        """
    raise NotImplementedError()