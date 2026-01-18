from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpHealthCheck(_messages.Message):
    """Represents a legacy HTTP Health Check resource. Legacy HTTP health
  checks are now only required by target pool-based network load balancers.
  For all other load balancers, including backend service-based network load
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
    host: The value of the host header in the HTTP health check request. If
      left empty (default value), the public IP on behalf of which this health
      check is performed will be used.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of the resource. Always compute#httpHealthCheck
      for HTTP health checks.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    port: The TCP port number for the HTTP health check request. The default
      value is 80.
    requestPath: The request path of the HTTP health check request. The
      default value is /. This field does not support query parameters. Must
      comply with RFC3986.
    selfLink: [Output Only] Server-defined URL for the resource.
    timeoutSec: How long (in seconds) to wait before claiming failure. The
      default value is 5 seconds. It is invalid for timeoutSec to have greater
      value than checkIntervalSec.
    unhealthyThreshold: A so-far healthy instance will be marked unhealthy
      after this many consecutive failures. The default value is 2.
  """
    checkIntervalSec = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    creationTimestamp = _messages.StringField(2)
    description = _messages.StringField(3)
    healthyThreshold = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    host = _messages.StringField(5)
    id = _messages.IntegerField(6, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(7, default='compute#httpHealthCheck')
    name = _messages.StringField(8)
    port = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    requestPath = _messages.StringField(10)
    selfLink = _messages.StringField(11)
    timeoutSec = _messages.IntegerField(12, variant=_messages.Variant.INT32)
    unhealthyThreshold = _messages.IntegerField(13, variant=_messages.Variant.INT32)