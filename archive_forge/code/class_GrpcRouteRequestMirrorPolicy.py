from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteRequestMirrorPolicy(_messages.Message):
    """Specifies the policy on how requests are mirrored to a separate mirrored
  destination service. The proxy does not wait for responses from the mirrored
  service. Prior to sending traffic to the mirrored service, the
  host/authority header is suffixed with -shadow.

  Fields:
    destination: The destination the requests will be mirrored to. The weight
      of the destination will be ignored.
  """
    destination = _messages.MessageField('GrpcRouteDestination', 1)