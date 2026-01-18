from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingDestination(_messages.Message):
    """Configuration of a specific billing destination (Currently only support
  bill against consumer project).

  Fields:
    metrics: Names of the metrics to report to this billing destination. Each
      name must be defined in Service.metrics section.
    monitoredResource: The monitored resource type. The type must be defined
      in Service.monitored_resources section.
  """
    metrics = _messages.StringField(1, repeated=True)
    monitoredResource = _messages.StringField(2)