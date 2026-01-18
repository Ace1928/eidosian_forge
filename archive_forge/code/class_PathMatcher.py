from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PathMatcher(_messages.Message):
    """The name of the `PathMatcher` to use to match the path portion of the
  URL if the HostRule matches the URL's host portion.

  Fields:
    description: Optional. A human-readable description of the resource.
    name: Required. The name to which this `PathMatcher` is referred by the
      HostRule.
    routeRules: Required. A list of RouteRule rules to match against.
      `RouteRule` rules support advanced routing behavior, and can match on
      paths, headers and query parameters, as well as status codes and HTTP
      methods. You must specify at least one rule, and can specify a maximum
      of 200 rules. `RouteRule` rules must not have duplicate priority values.
  """
    description = _messages.StringField(1)
    name = _messages.StringField(2)
    routeRules = _messages.MessageField('RouteRule', 3, repeated=True)