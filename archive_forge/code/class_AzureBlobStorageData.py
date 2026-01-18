from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AzureBlobStorageData(_messages.Message):
    """An AzureBlobStorageData resource can be a data source, but not a data
  sink. An AzureBlobStorageData resource represents one Azure container. The
  storage account determines the [Azure
  endpoint](https://docs.microsoft.com/en-us/azure/storage/common/storage-
  create-storage-account#storage-account-endpoints). In an
  AzureBlobStorageData resource, a blobs's name is the [Azure Blob Storage
  blob's key name](https://docs.microsoft.com/en-
  us/rest/api/storageservices/naming-and-referencing-containers--blobs--and-
  metadata#blob-names).

  Fields:
    azureCredentials: Required. Input only. Credentials used to authenticate
      API requests to Azure. For information on our data retention policy for
      user credentials, see [User credentials](/storage-transfer/docs/data-
      retention#user-credentials).
    container: Required. The container to transfer from the Azure Storage
      account.
    credentialsSecret: Optional. The Resource name of a secret in Secret
      Manager. The Azure SAS token must be stored in Secret Manager in JSON
      format: { "sas_token" : "SAS_TOKEN" } GoogleServiceAccount must be
      granted `roles/secretmanager.secretAccessor` for the resource. See
      [Configure access to a source: Microsoft Azure Blob Storage]
      (https://cloud.google.com/storage-transfer/docs/source-microsoft-
      azure#secret_manager) for more information. If `credentials_secret` is
      specified, do not specify azure_credentials. Format:
      `projects/{project_number}/secrets/{secret_name}`
    path: Root path to transfer objects. Must be an empty string or full path
      name that ends with a '/'. This field is treated as an object prefix. As
      such, it should generally not begin with a '/'.
    storageAccount: Required. The name of the Azure Storage account.
  """
    azureCredentials = _messages.MessageField('AzureCredentials', 1)
    container = _messages.StringField(2)
    credentialsSecret = _messages.StringField(3)
    path = _messages.StringField(4)
    storageAccount = _messages.StringField(5)