from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyDailyCycle(_messages.Message):
    """Time window specified for daily operations.

  Fields:
    daysInCycle: Defines a schedule with units measured in days. The value
      determines how many days pass between the start of each cycle.
    duration: [Output only] A predetermined duration for the window,
      automatically chosen to be the smallest possible in the given scenario.
    startTime: Start time of the window. This must be in UTC format that
      resolves to one of 00:00, 04:00, 08:00, 12:00, 16:00, or 20:00. For
      example, both 13:00-5 and 08:00 are valid.
  """
    daysInCycle = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    duration = _messages.StringField(2)
    startTime = _messages.StringField(3)