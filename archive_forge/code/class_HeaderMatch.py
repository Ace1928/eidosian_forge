from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HeaderMatch(_messages.Message):
    """The match conditions for HTTP request headers.

  Fields:
    exactMatch: Optional. The value of the header must exactly match contents
      of `exact_match`. Only one of `exact_match`, prefix_match, suffix_match,
      or present_match must be set.
    headerName: Required. The header name to match on. The `:method` pseudo-
      header may be used to match on the request HTTP method.
    invertMatch: Optional. If set to `false`, HeaderMatch is considered a
      match when the match criteria above are met. If set to `true`,
      `HeaderMatch` is considered a match when the match criteria above are
      not met. The default is `false`.
    prefixMatch: Optional. The value of the header must start with the
      contents of `prefix_match`. Only one of exact_match, `prefix_match`,
      suffix_match, or present_match must be set.
    presentMatch: Optional. A header with the contents of header_name must
      exist. The match takes place whether or not the request's header has a
      value. Only one of exact_match, prefix_match, suffix_match, or
      `present_match` must be set.
    suffixMatch: Optional. The value of the header must end with the contents
      of `suffix_match`. Only one of exact_match, prefix_match,
      `suffix_match`, or present_match must be set.
  """
    exactMatch = _messages.StringField(1)
    headerName = _messages.StringField(2)
    invertMatch = _messages.BooleanField(3)
    prefixMatch = _messages.StringField(4)
    presentMatch = _messages.BooleanField(5)
    suffixMatch = _messages.StringField(6)