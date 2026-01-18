from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2Probe(_messages.Message):
    """Probe describes a health check to be performed against a container to
  determine whether it is alive or ready to receive traffic.

  Fields:
    failureThreshold: Optional. Minimum consecutive failures for the probe to
      be considered failed after having succeeded. Defaults to 3. Minimum
      value is 1.
    grpc: Optional. GRPC specifies an action involving a gRPC port. Exactly
      one of httpGet, tcpSocket, or grpc must be specified.
    httpGet: Optional. HTTPGet specifies the http request to perform. Exactly
      one of httpGet, tcpSocket, or grpc must be specified.
    initialDelaySeconds: Optional. Number of seconds after the container has
      started before the probe is initiated. Defaults to 0 seconds. Minimum
      value is 0. Maximum value for liveness probe is 3600. Maximum value for
      startup probe is 240.
    periodSeconds: Optional. How often (in seconds) to perform the probe.
      Default to 10 seconds. Minimum value is 1. Maximum value for liveness
      probe is 3600. Maximum value for startup probe is 240. Must be greater
      or equal than timeout_seconds.
    tcpSocket: Optional. TCPSocket specifies an action involving a TCP port.
      Exactly one of httpGet, tcpSocket, or grpc must be specified.
    timeoutSeconds: Optional. Number of seconds after which the probe times
      out. Defaults to 1 second. Minimum value is 1. Maximum value is 3600.
      Must be smaller than period_seconds.
  """
    failureThreshold = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    grpc = _messages.MessageField('GoogleCloudRunV2GRPCAction', 2)
    httpGet = _messages.MessageField('GoogleCloudRunV2HTTPGetAction', 3)
    initialDelaySeconds = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    periodSeconds = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    tcpSocket = _messages.MessageField('GoogleCloudRunV2TCPSocketAction', 6)
    timeoutSeconds = _messages.IntegerField(7, variant=_messages.Variant.INT32)