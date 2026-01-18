from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudAuditLogsSource(_messages.Message):
    """A CloudAuditLogsSource object.

  Fields:
    apiVersion: The API version for this call such as
      "events.cloud.google.com/v1".
    kind: The kind of resource, in this case "CloudAuditLogsSource".
    metadata: Metadata associated with this CloudAuditLogsSource.
    spec: Spec defines the desired state of the CloudAuditLogsSource.
    status: Status represents the current state of the CloudAuditLogsSource.
      This data may be out of date. +optional
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    metadata = _messages.MessageField('ObjectMeta', 3)
    spec = _messages.MessageField('CloudAuditLogsSourceSpec', 4)
    status = _messages.MessageField('CloudAuditLogsSourceStatus', 5)