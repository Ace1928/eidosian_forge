from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlertingQueryStep(_messages.Message):
    """A query step defined as a set of alerting configuration options. This
  may not be used as the first step in a query.

  Fields:
    booleanCondition: A test representing the boolean value of a column.
    partitionColumns: Optional. The list of columns to GROUP BY in the
      generated SQL. NOTE: partition columns are not yet supported.
    stringCondition: A test representing a comparison against a string.
    thresholdCondition: A test representing a comparison against a threshold.
  """
    booleanCondition = _messages.MessageField('AlertingBooleanTest', 1)
    partitionColumns = _messages.StringField(2, repeated=True)
    stringCondition = _messages.MessageField('AlertingStringTest', 3)
    thresholdCondition = _messages.MessageField('AlertingThresholdTest', 4)