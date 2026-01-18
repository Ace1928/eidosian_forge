from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Constraints(_messages.Message):
    """Constraints to be applied while editing a schedule. These constraints
  ensure that `Upgrade` specific requirements are met.

  Fields:
    minHoursDay: Output only. Minimum number of hours must be allotted for the
      upgrade activities for each selected day. This is a minimum; the upgrade
      schedule can allot more hours for the given day.
    minHoursWeek: Output only. The minimum number of weekly hours must be
      allotted for the upgrade activities. This is just a minimum; the
      schedule can assign more weekly hours.
    rescheduleDateRange: Output only. Output Only. The user can only
      reschedule an upgrade that starts within this range.
  """
    minHoursDay = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minHoursWeek = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    rescheduleDateRange = _messages.MessageField('Interval', 3)