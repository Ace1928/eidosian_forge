from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteHeaderMatch(_messages.Message):
    """Specifies how to select a route rule based on HTTP request headers.

  Fields:
    exactMatch: The value of the header should match exactly the content of
      exact_match.
    header: The name of the HTTP header to match against.
    invertMatch: If specified, the match result will be inverted before
      checking. Default value is set to false.
    prefixMatch: The value of the header must start with the contents of
      prefix_match.
    presentMatch: A header with header_name must exist. The match takes place
      whether or not the header has a value.
    rangeMatch: If specified, the rule will match if the request header value
      is within the range.
    regexMatch: The value of the header must match the regular expression
      specified in regex_match. For regular expression grammar, please see:
      https://github.com/google/re2/wiki/Syntax
    suffixMatch: The value of the header must end with the contents of
      suffix_match.
  """
    exactMatch = _messages.StringField(1)
    header = _messages.StringField(2)
    invertMatch = _messages.BooleanField(3)
    prefixMatch = _messages.StringField(4)
    presentMatch = _messages.BooleanField(5)
    rangeMatch = _messages.MessageField('HttpRouteHeaderMatchIntegerRange', 6)
    regexMatch = _messages.StringField(7)
    suffixMatch = _messages.StringField(8)