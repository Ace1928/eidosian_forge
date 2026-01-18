from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClientsDeleteRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAzureClientsDeleteRequest object.

  Fields:
    allowMissing: If set to true, and the AzureClient resource is not found,
      the request will succeed but no action will be taken on the server and a
      completed Operation will be returned. Useful for idempotent deletion.
    name: Required. The resource name the AzureClient to delete. AzureClient
      names are formatted as `projects//locations//azureClients/`. See
      [Resource Names](https://cloud.google.com/apis/design/resource_names)
      for more details on Google Cloud resource names.
    validateOnly: If set, only validate the request, but do not actually
      delete the resource.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)