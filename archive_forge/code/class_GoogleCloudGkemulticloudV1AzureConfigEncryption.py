from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureConfigEncryption(_messages.Message):
    """Configuration related to config data encryption. Azure VM bootstrap
  secret is envelope encrypted with the provided key vault key.

  Fields:
    keyId: Required. The ARM ID of the Azure Key Vault key to encrypt /
      decrypt config data. For example: `/subscriptions//resourceGroups//provi
      ders/Microsoft.KeyVault/vaults//keys/`
    publicKey: Optional. RSA key of the Azure Key Vault public key to use for
      encrypting the data. This key must be formatted as a PEM-encoded
      SubjectPublicKeyInfo (RFC 5280) in ASN.1 DER form. The string must be
      comprised of a single PEM block of type "PUBLIC KEY".
  """
    keyId = _messages.StringField(1)
    publicKey = _messages.StringField(2)