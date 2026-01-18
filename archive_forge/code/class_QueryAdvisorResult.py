from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryAdvisorResult(_messages.Message):
    """Output of query advisor analysis.

  Fields:
    indexAdvice: Optional. Index Recommendation for a query. This is an
      optional field and the recommendation will only be available when the
      recommendation guarantees significant improvement in query performance.
  """
    indexAdvice = _messages.MessageField('IndexAdvice', 1, repeated=True)