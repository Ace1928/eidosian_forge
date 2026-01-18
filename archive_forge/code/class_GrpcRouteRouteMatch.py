from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteRouteMatch(_messages.Message):
    """Criteria for matching traffic. A RouteMatch will be considered to match
  when all supplied fields match.

  Fields:
    headers: Optional. Specifies a collection of headers to match.
    method: Optional. A gRPC method to match against. If this field is empty
      or omitted, will match all methods.
  """
    headers = _messages.MessageField('GrpcRouteHeaderMatch', 1, repeated=True)
    method = _messages.MessageField('GrpcRouteMethodMatch', 2)