from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityRuleTableConditionExpectation(_messages.Message):
    """Evaluates whether the provided expression is true.The SQL expression
  needs to use BigQuery standard SQL syntax and should produce a scalar
  boolean result.Example: MIN(col1) >= 0

  Fields:
    sqlExpression: Optional. The SQL expression.
  """
    sqlExpression = _messages.StringField(1)