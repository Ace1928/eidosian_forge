from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsRouteRouteDestination(_messages.Message):
    """Describe the destination for traffic to be routed to.

  Fields:
    serviceName: Required. The URL of a BackendService to route traffic to.
    weight: Optional. Specifies the proportion of requests forwareded to the
      backend referenced by the service_name field. This is computed as: -
      weight/Sum(weights in destinations) Weights in all destinations does not
      need to sum up to 100.
  """
    serviceName = _messages.StringField(1)
    weight = _messages.IntegerField(2, variant=_messages.Variant.INT32)