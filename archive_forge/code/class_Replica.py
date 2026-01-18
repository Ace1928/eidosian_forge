from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Replica(_messages.Message):
    """Represents a Replica for this Secret.

  Fields:
    customerManagedEncryption: Optional. The customer-managed encryption
      configuration of the User-Managed Replica. If no configuration is
      provided, Google-managed default encryption is used. Updates to the
      Secret encryption configuration only apply to SecretVersions added
      afterwards. They do not apply retroactively to existing SecretVersions.
    location: The canonical IDs of the location to replicate data. For
      example: `"us-east1"`.
  """
    customerManagedEncryption = _messages.MessageField('CustomerManagedEncryption', 1)
    location = _messages.StringField(2)