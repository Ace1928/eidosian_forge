from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionSpec(_messages.Message):
    """Represents a customer-managed encryption key spec that can be applied to
  a resource.

  Fields:
    kmsKeyName: Required. The resource name of customer-managed encryption key
      that is used to secure a resource and its sub-resources. Only the key in
      the same location as this Dataset is allowed to be used for encryption.
      Format is: `projects/{project}/locations/{location}/keyRings/{keyRing}/c
      ryptoKeys/{key}`
  """
    kmsKeyName = _messages.StringField(1)