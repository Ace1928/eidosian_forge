from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AggregateValueThreshold(_messages.Message):
    """A threshold condition that compares an aggregation to a threshold.

  Fields:
    aggregateColumn: Required. The column to provide aggregation on for
      comparison.
    aggregation: Required. The aggregation config that will be applied to the
      provided column.
  """
    aggregateColumn = _messages.StringField(1)
    aggregation = _messages.MessageField('QueryStepAggregation', 2)