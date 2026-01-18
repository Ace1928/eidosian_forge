from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudauditlogssourcesCreateRequest(_messages.Message):
    """A AnthoseventsNamespacesCloudauditlogssourcesCreateRequest object.

  Fields:
    cloudAuditLogsSource: A CloudAuditLogsSource resource to be passed as the
      request body.
    parent: The namespace name.
  """
    cloudAuditLogsSource = _messages.MessageField('CloudAuditLogsSource', 1)
    parent = _messages.StringField(2, required=True)