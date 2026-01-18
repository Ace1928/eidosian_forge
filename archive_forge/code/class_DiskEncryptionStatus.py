from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DiskEncryptionStatus(_messages.Message):
    """Disk encryption status for an instance.

  Fields:
    kind: This is always `sql#diskEncryptionStatus`.
    kmsKeyVersionName: KMS key version used to encrypt the Cloud SQL instance
      resource
  """
    kind = _messages.StringField(1)
    kmsKeyVersionName = _messages.StringField(2)