from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteRouteRule(_messages.Message):
    """Specifies how to match traffic and how to route traffic when traffic is
  matched.

  Fields:
    action: The detailed rule defining how to route matched traffic.
    matches: A list of matches define conditions used for matching the rule
      against incoming HTTP requests. Each match is independent, i.e. this
      rule will be matched if ANY one of the matches is satisfied. If no
      matches field is specified, this rule will unconditionally match
      traffic. If a default rule is desired to be configured, add a rule with
      no matches specified to the end of the rules list.
  """
    action = _messages.MessageField('HttpRouteRouteAction', 1)
    matches = _messages.MessageField('HttpRouteRouteMatch', 2, repeated=True)