from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteURLRewrite(_messages.Message):
    """The specification to modify the URL of the request, prior to forwarding
  the request to the destination.

  Fields:
    hostRewrite: Prior to forwarding the request to the selected destination,
      the requests host header is replaced by this value.
    pathPrefixRewrite: Prior to forwarding the request to the selected
      destination, the matching portion of the requests path is replaced by
      this value.
  """
    hostRewrite = _messages.StringField(1)
    pathPrefixRewrite = _messages.StringField(2)