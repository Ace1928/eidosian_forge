from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualitySpecPostScanActionsRecipients(_messages.Message):
    """The individuals or groups who are designated to receive notifications
  upon triggers.

  Fields:
    emails: Optional. The email recipients who will receive the
      DataQualityScan results report.
  """
    emails = _messages.StringField(1, repeated=True)