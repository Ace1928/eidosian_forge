from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryOverride(_messages.Message):
    """QueryOverride. Query message defines query override for HTTP targets.

  Fields:
    queryParams: The query parameters (e.g., qparam1=123&qparam2=456). Default
      is an empty string.
  """
    queryParams = _messages.StringField(1)