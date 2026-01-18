import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class DynamicStub(abc.ABC):
    """Affords RPC invocation via attributes corresponding to afforded methods.

    Instances of this type may be scoped to a single group so that attribute
    access is unambiguous.

    Instances of this type respond to attribute access as follows: if the
    requested attribute is the name of a unary-unary method, the value of the
    attribute will be a UnaryUnaryMultiCallable with which to invoke an RPC; if
    the requested attribute is the name of a unary-stream method, the value of the
    attribute will be a UnaryStreamMultiCallable with which to invoke an RPC; if
    the requested attribute is the name of a stream-unary method, the value of the
    attribute will be a StreamUnaryMultiCallable with which to invoke an RPC; and
    if the requested attribute is the name of a stream-stream method, the value of
    the attribute will be a StreamStreamMultiCallable with which to invoke an RPC.
    """