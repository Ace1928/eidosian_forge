from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataScanExecutionSpec(_messages.Message):
    """DataScan execution settings.

  Fields:
    field: Immutable. The unnested field (of type Date or Timestamp) that
      contains values which monotonically increase over time.If not specified,
      a data scan will run for all data in the table.
    trigger: Optional. Spec related to how often and when a scan should be
      triggered.If not specified, the default is OnDemand, which means the
      scan will not run until the user calls RunDataScan API.
  """
    field = _messages.StringField(1)
    trigger = _messages.MessageField('GoogleCloudDataplexV1Trigger', 2)