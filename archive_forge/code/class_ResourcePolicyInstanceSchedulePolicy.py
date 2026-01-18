from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyInstanceSchedulePolicy(_messages.Message):
    """An InstanceSchedulePolicy specifies when and how frequent certain
  operations are performed on the instance.

  Fields:
    expirationTime: The expiration time of the schedule. The timestamp is an
      RFC3339 string.
    startTime: The start time of the schedule. The timestamp is an RFC3339
      string.
    timeZone: Specifies the time zone to be used in interpreting
      Schedule.schedule. The value of this field must be a time zone name from
      the tz database: https://wikipedia.org/wiki/Tz_database.
    vmStartSchedule: Specifies the schedule for starting instances.
    vmStopSchedule: Specifies the schedule for stopping instances.
  """
    expirationTime = _messages.StringField(1)
    startTime = _messages.StringField(2)
    timeZone = _messages.StringField(3)
    vmStartSchedule = _messages.MessageField('ResourcePolicyInstanceSchedulePolicySchedule', 4)
    vmStopSchedule = _messages.MessageField('ResourcePolicyInstanceSchedulePolicySchedule', 5)