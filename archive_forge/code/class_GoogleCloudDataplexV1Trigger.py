from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Trigger(_messages.Message):
    """DataScan scheduling and trigger settings.

  Fields:
    onDemand: The scan runs once via RunDataScan API.
    schedule: The scan is scheduled to run periodically.
  """
    onDemand = _messages.MessageField('GoogleCloudDataplexV1TriggerOnDemand', 1)
    schedule = _messages.MessageField('GoogleCloudDataplexV1TriggerSchedule', 2)