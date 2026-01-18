from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingDestination(_messages.Message):
    """Configuration of a specific logging destination (the producer project or
  the consumer project).

  Fields:
    logs: Names of the logs to be sent to this destination. Each name must be
      defined in the Service.logs section.
    monitoredResource: The monitored resource type. The type must be defined
      in Service.monitored_resources section.
  """
    logs = _messages.StringField(1, repeated=True)
    monitoredResource = _messages.StringField(2)