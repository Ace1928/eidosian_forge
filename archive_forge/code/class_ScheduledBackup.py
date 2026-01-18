from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScheduledBackup(_messages.Message):
    """This specifies the configuration of scheduled backup.

  Fields:
    backupLocation: Optional. A Cloud Storage URI of a folder, in the format
      gs:///. A sub-folder containing backup files will be stored below it.
    cronSchedule: Optional. The scheduled interval in Cron format, see
      https://en.wikipedia.org/wiki/Cron The default is empty: scheduled
      backup is not enabled. Must be specified to enable scheduled backups.
    enabled: Optional. Defines whether the scheduled backup is enabled. The
      default value is false.
    latestBackup: Output only. The details of the latest scheduled backup.
    nextScheduledTime: Output only. The time when the next backups execution
      is scheduled to start.
    timeZone: Optional. Specifies the time zone to be used when interpreting
      cron_schedule. Must be a time zone name from the time zone database
      (https://en.wikipedia.org/wiki/List_of_tz_database_time_zones), e.g.
      America/Los_Angeles or Africa/Abidjan. If left unspecified, the default
      is UTC.
  """
    backupLocation = _messages.StringField(1)
    cronSchedule = _messages.StringField(2)
    enabled = _messages.BooleanField(3)
    latestBackup = _messages.MessageField('LatestBackup', 4)
    nextScheduledTime = _messages.StringField(5)
    timeZone = _messages.StringField(6)