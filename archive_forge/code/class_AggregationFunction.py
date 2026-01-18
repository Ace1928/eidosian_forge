from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AggregationFunction(_messages.Message):
    """Preview: An identifier for an aggregation function. Aggregation
  functions are SQL functions that group or transform data from multiple
  points to a single point. This is a preview feature and may be subject to
  change before final release.

  Fields:
    parameters: Optional. Parameters applied to the aggregation function. Only
      used for functions that require them.
    type: Required. The type of aggregation function, must be one of the
      following: "none" - no function. "percentile" - APPROX_QUANTILES() - 1
      parameter numeric value "average" - AVG() "count" - COUNT() "count-
      distinct" - COUNT(DISTINCT) "count-distinct-approx" -
      APPROX_COUNT_DISTINCT() "max" - MAX() "min" - MIN() "sum" - SUM()
  """
    parameters = _messages.MessageField('Parameter', 1, repeated=True)
    type = _messages.StringField(2)