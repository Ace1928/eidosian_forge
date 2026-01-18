from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RangePartitioning(_messages.Message):
    """A RangePartitioning object.

  Messages:
    RangeValue: [Experimental] Defines the ranges for range partitioning.

  Fields:
    field: Required. [Experimental] The table is partitioned by this field.
      The field must be a top-level NULLABLE/REQUIRED field. The only
      supported type is INTEGER/INT64.
    range: [Experimental] Defines the ranges for range partitioning.
  """

    class RangeValue(_messages.Message):
        """[Experimental] Defines the ranges for range partitioning.

    Fields:
      end: [Experimental] The end of range partitioning, exclusive.
      interval: [Experimental] The width of each interval.
      start: [Experimental] The start of range partitioning, inclusive.
    """
        end = _messages.IntegerField(1)
        interval = _messages.IntegerField(2)
        start = _messages.IntegerField(3)
    field = _messages.StringField(1)
    range = _messages.MessageField('RangeValue', 2)