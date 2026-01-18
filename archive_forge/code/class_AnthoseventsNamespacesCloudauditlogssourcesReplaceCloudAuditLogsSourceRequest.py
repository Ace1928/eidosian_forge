from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudauditlogssourcesReplaceCloudAuditLogsSourceRequest(_messages.Message):
    """A AnthoseventsNamespacesCloudauditlogssourcesReplaceCloudAuditLogsSource
  Request object.

  Fields:
    cloudAuditLogsSource: A CloudAuditLogsSource resource to be passed as the
      request body.
    name: The name of the cloudauditlogssource being retrieved. If needed,
      replace {namespace_id} with the project ID.
  """
    cloudAuditLogsSource = _messages.MessageField('CloudAuditLogsSource', 1)
    name = _messages.StringField(2, required=True)