from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndpointProtocolValueValuesEnum(_messages.Enum):
    """The protocol to use for the metastore service endpoint. If
    unspecified, defaults to THRIFT.

    Values:
      ENDPOINT_PROTOCOL_UNSPECIFIED: The protocol is not set.
      THRIFT: Use the legacy Apache Thrift protocol for the metastore service
        endpoint.
      GRPC: Use the modernized gRPC protocol for the metastore service
        endpoint.
    """
    ENDPOINT_PROTOCOL_UNSPECIFIED = 0
    THRIFT = 1
    GRPC = 2