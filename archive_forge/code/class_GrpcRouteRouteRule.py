from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteRouteRule(_messages.Message):
    """Describes how to route traffic.

  Fields:
    action: Required. A detailed rule defining how to route traffic. This
      field is required.
    matches: Optional. Matches define conditions used for matching the rule
      against incoming gRPC requests. Each match is independent, i.e. this
      rule will be matched if ANY one of the matches is satisfied. If no
      matches field is specified, this rule will unconditionally match
      traffic.
  """
    action = _messages.MessageField('GrpcRouteRouteAction', 1)
    matches = _messages.MessageField('GrpcRouteRouteMatch', 2, repeated=True)