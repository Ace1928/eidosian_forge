from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryParameterMatcher(_messages.Message):
    """The match conditions for URI query parameters.

  Fields:
    exactMatch: Optional. The QueryParameterMatcher matches if the value of
      the parameter exactly matches the contents of `exact_match`. The match
      value must be between 1 and 64 characters long (inclusive). Only one of
      present_match or `exact_match` must be set.
    name: Required. The name of the query parameter to match. The query
      parameter must exist in the request; if it doesn't, the request match
      fails. The name must be specified and be between 1 and 32 characters
      long (inclusive).
    presentMatch: Optional. Specifies that the QueryParameterMatcher matches
      if the request contains the query parameter. The match can succeed as
      long as the request contains the query parameter, regardless of whether
      the parameter has a value or not. Only one of `present_match` or
      exact_match must be set.
  """
    exactMatch = _messages.StringField(1)
    name = _messages.StringField(2)
    presentMatch = _messages.BooleanField(3)