from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExplainOptions(_messages.Message):
    """Explain options for the query.

  Fields:
    analyze: Optional. Whether to execute this query. When false (the
      default), the query will be planned, returning only metrics from the
      planning stages. When true, the query will be planned and executed,
      returning the full query results along with both planning and execution
      stage metrics.
  """
    analyze = _messages.BooleanField(1)