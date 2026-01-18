from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DiskEncryptionConfiguration(_messages.Message):
    """Disk encryption configuration for an instance.

  Fields:
    kind: This is always `sql#diskEncryptionConfiguration`.
    kmsKeyName: Resource name of KMS key for disk encryption
  """
    kind = _messages.StringField(1)
    kmsKeyName = _messages.StringField(2)