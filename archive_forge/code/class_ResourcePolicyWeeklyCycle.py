from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyWeeklyCycle(_messages.Message):
    """Time window specified for weekly operations.

  Fields:
    dayOfWeeks: Up to 7 intervals/windows, one for each day of the week.
  """
    dayOfWeeks = _messages.MessageField('ResourcePolicyWeeklyCycleDayOfWeek', 1, repeated=True)