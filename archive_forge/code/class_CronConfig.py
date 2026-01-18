from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CronConfig(_messages.Message):
    """CronConfig describes the configuration of a trigger that creates a build
  whenever a Cloud Scheduler event is received.

  Fields:
    enterpriseConfigResource: The GitHub Enterprise config resource name that
      is associated with this installation.
    schedule: Required. Describes the schedule on which the job will be
      executed. The schedule can be either of the following types: *
      [Crontab](http://en.wikipedia.org/wiki/Cron#Overview) * English-like
      [schedule](https://cloud.google.com/scheduler/docs/configuring/cron-job-
      schedules)
    timeZone: Specifies the time zone to be used in interpreting the schedule.
      The value of this field must be a time zone name from the [tz database]
      (http://en.wikipedia.org/wiki/Tz_database). Note that some time zones
      include a provision for daylight savings time. The rules for daylight
      saving time are determined by the chosen tz. For UTC use the string
      "utc". If a time zone is not specified, the default will be in UTC (also
      known as GMT).
  """
    enterpriseConfigResource = _messages.StringField(1)
    schedule = _messages.StringField(2)
    timeZone = _messages.StringField(3)