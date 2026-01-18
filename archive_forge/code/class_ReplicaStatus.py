from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplicaStatus(_messages.Message):
    """Describes the status of a user-managed replica for the SecretVersion.

  Fields:
    customerManagedEncryption: Output only. The customer-managed encryption
      status of the SecretVersion. Only populated if customer-managed
      encryption is used.
    location: Output only. The canonical ID of the replica location. For
      example: `"us-east1"`.
  """
    customerManagedEncryption = _messages.MessageField('CustomerManagedEncryptionStatus', 1)
    location = _messages.StringField(2)