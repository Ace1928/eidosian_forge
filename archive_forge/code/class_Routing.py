from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Routing(_messages.Message):
    """Defines how requests are routed, modified, cached, and which origin the
  content is filled from.

  Fields:
    hostRules: Required. A list of HostRule rules to match against.
      `RouteRule` rules support advanced routing behavior, and can match on
      paths, headers and query parameters, as well as status codes and HTTP
      methods. You can specify up to 10 host rules.
    pathMatchers: Required. A list of PathMatcher values referenced by name by
      HostRule values. `PathMatcher` is used to match the path portion of the
      URL when a`HostRule` value matches the URL's host portion. You can
      specify up to 10 path matchers.
  """
    hostRules = _messages.MessageField('HostRule', 1, repeated=True)
    pathMatchers = _messages.MessageField('PathMatcher', 2, repeated=True)