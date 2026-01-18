from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringDestination(_messages.Message):
    """Configuration of a specific monitoring destination (the producer project
  or the consumer project).

  Fields:
    metrics: Names of the metrics to report to this monitoring destination.
      Each name must be defined in Service.metrics section.
    monitoredResource: The monitored resource type. The type must be defined
      in Service.monitored_resources section.
  """
    metrics = _messages.StringField(1, repeated=True)
    monitoredResource = _messages.StringField(2)