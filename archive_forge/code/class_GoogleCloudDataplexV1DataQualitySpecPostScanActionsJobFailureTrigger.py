from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualitySpecPostScanActionsJobFailureTrigger(_messages.Message):
    """This trigger is triggered when the scan job itself fails, regardless of
  the result.
  """