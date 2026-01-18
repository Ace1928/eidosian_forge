from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GaugeView(_messages.Message):
    """A gauge chart shows where the current value sits within a pre-defined
  range. The upper and lower bounds should define the possible range of values
  for the scorecard's query (inclusive).

  Fields:
    lowerBound: The lower bound for this gauge chart. The value of the chart
      should always be greater than or equal to this.
    upperBound: The upper bound for this gauge chart. The value of the chart
      should always be less than or equal to this.
  """
    lowerBound = _messages.FloatField(1)
    upperBound = _messages.FloatField(2)