from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteRouteMatch(_messages.Message):
    """RouteMatch defines specifications used to match requests. If multiple
  match types are set, this RouteMatch will match if ALL type of matches are
  matched.

  Fields:
    fullPathMatch: The HTTP request path value should exactly match this
      value. Only one of full_path_match, prefix_match, or regex_match should
      be used.
    headers: Specifies a list of HTTP request headers to match against. ALL of
      the supplied headers must be matched.
    ignoreCase: Specifies if prefix_match and full_path_match matches are case
      sensitive. The default value is false.
    prefixMatch: The HTTP request path value must begin with specified
      prefix_match. prefix_match must begin with a /. Only one of
      full_path_match, prefix_match, or regex_match should be used.
    queryParameters: Specifies a list of query parameters to match against.
      ALL of the query parameters must be matched.
    regexMatch: The HTTP request path value must satisfy the regular
      expression specified by regex_match after removing any query parameters
      and anchor supplied with the original URL. For regular expression
      grammar, please see https://github.com/google/re2/wiki/Syntax Only one
      of full_path_match, prefix_match, or regex_match should be used.
  """
    fullPathMatch = _messages.StringField(1)
    headers = _messages.MessageField('HttpRouteHeaderMatch', 2, repeated=True)
    ignoreCase = _messages.BooleanField(3)
    prefixMatch = _messages.StringField(4)
    queryParameters = _messages.MessageField('HttpRouteQueryParameterMatch', 5, repeated=True)
    regexMatch = _messages.StringField(6)