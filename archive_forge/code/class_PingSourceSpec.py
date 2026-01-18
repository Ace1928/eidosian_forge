from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PingSourceSpec(_messages.Message):
    """The desired state of the PingSource.

  Fields:
    ceOverrides: CloudEventOverrides defines overrides to control the output
      format and modifications of the event sent to the sink.
    jsonData: JsonData is json encoded data used as the body of the event
      posted to the sink. Default is empty. If set, datacontenttype will also
      be set to "application/json".
    schedule: Schedule is the cronjob schedule. Defaults to `* * * * *`.
    sink: Sink is a reference to an object that will resolve to a uri to use
      as the sink.
    timezone: Timezone modifies the actual time relative to the specified
      timezone. Defaults to the system time zone. More general information
      about time zones: https://www.iana.org/time-zones List of valid timezone
      values: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
  """
    ceOverrides = _messages.MessageField('CloudEventOverrides', 1)
    jsonData = _messages.StringField(2)
    schedule = _messages.StringField(3)
    sink = _messages.MessageField('Destination', 4)
    timezone = _messages.StringField(5)