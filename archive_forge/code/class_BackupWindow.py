from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupWindow(_messages.Message):
    """`BackupWindow` defines a window of the day during which backup jobs will
  run.

  Fields:
    endHourOfDay: Required. The hour of day (1-24) when the window end for
      e.g. if value of end hour of day is 10 that mean backup window end time
      is 10:00. End hour of day should be greater than start hour of day. 0 <=
      start_hour_of_day < end_hour_of_day <= 24 End hour of day is not include
      in backup window that mean if end_hour_of_day= 10 jobs should start
      before 10:00.
    endTime: Optional. TODO b/325560313: Deprecated and field will be removed
      after UI integration change. The end time of the window in which to pick
      backup jobs to run.
    startHourOfDay: Required. The hour of day (0-23) when the window starts
      for e.g. if value of start hour of day is 6 that mean backup window
      start at 6:00.
    startTime: Optional. TODO b/325560313: Deprecated and field will be
      removed after UI integration change. The start time of the window in
      which to pick backup jobs to run.
  """
    endHourOfDay = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    endTime = _messages.MessageField('TimeOfDay', 2)
    startHourOfDay = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    startTime = _messages.MessageField('TimeOfDay', 4)