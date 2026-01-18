from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContinuousBackupInfo(_messages.Message):
    """ContinuousBackupInfo describes the continuous backup properties of a
  cluster.

  Enums:
    ScheduleValueListEntryValuesEnum:

  Fields:
    earliestRestorableTime: Output only. The earliest restorable time that can
      be restored to. Output only field.
    enabledTime: Output only. When ContinuousBackup was most recently enabled.
      Set to null if ContinuousBackup is not enabled.
    encryptionInfo: Output only. The encryption information for the WALs and
      backups required for ContinuousBackup.
    schedule: Output only. Days of the week on which a continuous backup is
      taken. Output only field. Ignored if passed into the request.
  """

    class ScheduleValueListEntryValuesEnum(_messages.Enum):
        """ScheduleValueListEntryValuesEnum enum type.

    Values:
      DAY_OF_WEEK_UNSPECIFIED: The day of the week is unspecified.
      MONDAY: Monday
      TUESDAY: Tuesday
      WEDNESDAY: Wednesday
      THURSDAY: Thursday
      FRIDAY: Friday
      SATURDAY: Saturday
      SUNDAY: Sunday
    """
        DAY_OF_WEEK_UNSPECIFIED = 0
        MONDAY = 1
        TUESDAY = 2
        WEDNESDAY = 3
        THURSDAY = 4
        FRIDAY = 5
        SATURDAY = 6
        SUNDAY = 7
    earliestRestorableTime = _messages.StringField(1)
    enabledTime = _messages.StringField(2)
    encryptionInfo = _messages.MessageField('EncryptionInfo', 3)
    schedule = _messages.EnumField('ScheduleValueListEntryValuesEnum', 4, repeated=True)