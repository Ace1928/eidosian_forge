from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpsHealthCheck(_messages.Message):
    """Represents a legacy HTTPS Health Check resource. Legacy HTTPS health
  checks have been deprecated. If you are using a target pool-based network
  load balancer, you must use a legacy HTTP (not HTTPS) health check. For all
  other load balancers, including backend service-based network load
  balancers, and for managed instance group auto-healing, you must use modern
  (non-legacy) health checks. For more information, see Health checks overview
  .

  Fields:
    checkIntervalSec: How often (in seconds) to send a health check. The
      default value is 5 seconds.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    healthyThreshold: A so-far unhealthy instance will be marked healthy after
      this many consecutive successes. The default value is 2.
    host: The value of the host header in the HTTPS health check request. If
      left empty (default value), the public IP on behalf of which this health
      check is performed will be used.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: Type of the resource.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    port: The TCP port number for the HTTPS health check request. The default
      value is 443.
    requestPath: The request path of the HTTPS health check request. The
      default value is "/". Must comply with RFC3986.
    selfLink: [Output Only] Server-defined URL for the resource.
    timeoutSec: How long (in seconds) to wait before claiming failure. The
      default value is 5 seconds. It is invalid for timeoutSec to have a
      greater value than checkIntervalSec.
    unhealthyThreshold: A so-far healthy instance will be marked unhealthy
      after this many consecutive failures. The default value is 2.
  """
    checkIntervalSec = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    creationTimestamp = _messages.StringField(2)
    description = _messages.StringField(3)
    healthyThreshold = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    host = _messages.StringField(5)
    id = _messages.IntegerField(6, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(7, default='compute#httpsHealthCheck')
    name = _messages.StringField(8)
    port = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    requestPath = _messages.StringField(10)
    selfLink = _messages.StringField(11)
    timeoutSec = _messages.IntegerField(12, variant=_messages.Variant.INT32)
    unhealthyThreshold = _messages.IntegerField(13, variant=_messages.Variant.INT32)