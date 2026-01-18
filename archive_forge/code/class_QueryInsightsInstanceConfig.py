from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryInsightsInstanceConfig(_messages.Message):
    """QueryInsights Instance specific configuration.

  Fields:
    queryPlansPerMinute: Number of query execution plans captured by Insights
      per minute for all queries combined. The default value is 5. Any integer
      between 0 and 20 is considered valid.
    queryStringLength: Query string length. The default value is 1024. Any
      integer between 256 and 4500 is considered valid.
    recordApplicationTags: Record application tags for an instance. This flag
      is turned "on" by default.
    recordClientAddress: Record client address for an instance. Client address
      is PII information. This flag is turned "on" by default.
  """
    queryPlansPerMinute = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    queryStringLength = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    recordApplicationTags = _messages.BooleanField(3)
    recordClientAddress = _messages.BooleanField(4)