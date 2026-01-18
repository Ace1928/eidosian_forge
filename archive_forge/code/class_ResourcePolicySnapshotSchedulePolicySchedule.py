from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicySnapshotSchedulePolicySchedule(_messages.Message):
    """A schedule for disks where the schedueled operations are performed.

  Fields:
    dailySchedule: A ResourcePolicyDailyCycle attribute.
    hourlySchedule: A ResourcePolicyHourlyCycle attribute.
    weeklySchedule: A ResourcePolicyWeeklyCycle attribute.
  """
    dailySchedule = _messages.MessageField('ResourcePolicyDailyCycle', 1)
    hourlySchedule = _messages.MessageField('ResourcePolicyHourlyCycle', 2)
    weeklySchedule = _messages.MessageField('ResourcePolicyWeeklyCycle', 3)