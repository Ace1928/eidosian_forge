from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2GRPCAction(_messages.Message):
    """GRPCAction describes an action involving a GRPC port.

  Fields:
    port: Optional. Port number of the gRPC service. Number must be in the
      range 1 to 65535. If not specified, defaults to the exposed port of the
      container, which is the value of container.ports[0].containerPort.
    service: Optional. Service is the name of the service to place in the gRPC
      HealthCheckRequest (see
      https://github.com/grpc/grpc/blob/master/doc/health-checking.md ). If
      this is not specified, the default behavior is defined by gRPC.
  """
    port = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    service = _messages.StringField(2)