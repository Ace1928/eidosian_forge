from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TcpRouteRouteRule(_messages.Message):
    """Specifies how to match traffic and how to route traffic when traffic is
  matched.

  Fields:
    action: Required. The detailed rule defining how to route matched traffic.
    matches: Optional. RouteMatch defines the predicate used to match requests
      to a given action. Multiple match types are "OR"ed for evaluation. If no
      routeMatch field is specified, this rule will unconditionally match
      traffic.
  """
    action = _messages.MessageField('TcpRouteRouteAction', 1)
    matches = _messages.MessageField('TcpRouteRouteMatch', 2, repeated=True)