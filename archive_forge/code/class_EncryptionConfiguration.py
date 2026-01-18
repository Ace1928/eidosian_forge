from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionConfiguration(_messages.Message):
    """Represents the encryption configuration for a transfer.

  Fields:
    kmsKeyName: The name of the KMS key used for encrypting BigQuery data.
  """
    kmsKeyName = _messages.StringField(1)