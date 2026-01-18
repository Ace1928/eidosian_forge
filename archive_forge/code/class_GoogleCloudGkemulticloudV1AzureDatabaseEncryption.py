from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureDatabaseEncryption(_messages.Message):
    """Configuration related to application-layer secrets encryption. Anthos
  clusters on Azure encrypts your Kubernetes data at rest in etcd using Azure
  Key Vault.

  Fields:
    keyId: Required. The ARM ID of the Azure Key Vault key to encrypt /
      decrypt data. For example: `/subscriptions//resourceGroups//providers/Mi
      crosoft.KeyVault/vaults//keys/` Encryption will always take the latest
      version of the key and hence specific version is not supported.
  """
    keyId = _messages.StringField(1)