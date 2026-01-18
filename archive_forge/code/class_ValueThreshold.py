from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValueThreshold(_messages.Message):
    """A threshold condition that compares a value to a threshold.

  Fields:
    trigger: Optional. The number/percent of rows that must exceed the
      threshold in order for this result set (partition set) to be considered
      in violation. If unspecified, then the result set (partition set) will
      be in violation when a single row violates the threshold.
    valueColumn: Required. The column to compare the threshold against.
  """
    trigger = _messages.MessageField('AlertingTrigger', 1)
    valueColumn = _messages.StringField(2)