from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MessageSubscriptionRouteRule(_messages.Message):
    """Specifies how to match traffic and how to route traffic when traffic is
  matched.

  Fields:
    action: Required. The action to take when the rule matches.
    celMatches: Optional. A list of CEL match expressions used for matching
      the rule against incoming messages. Each match is independent, i.e. this
      rule will be matched if ANY one of the matches is satisfied. If no match
      is specified, this rule will not match any traffic.
  """
    action = _messages.MessageField('MessageSubscriptionRouteAction', 1)
    celMatches = _messages.MessageField('MessageSubscriptionCelMatch', 2, repeated=True)