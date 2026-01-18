from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlServerAuditConfig(_messages.Message):
    """SQL Server specific audit configuration.

  Fields:
    bucket: The name of the destination bucket (e.g., gs://mybucket).
    kind: This is always sql#sqlServerAuditConfig
    retentionInterval: How long to keep generated audit files.
    uploadInterval: How often to upload generated audit files.
  """
    bucket = _messages.StringField(1)
    kind = _messages.StringField(2)
    retentionInterval = _messages.StringField(3)
    uploadInterval = _messages.StringField(4)