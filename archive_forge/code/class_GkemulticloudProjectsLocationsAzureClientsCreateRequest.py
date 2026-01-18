from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClientsCreateRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAzureClientsCreateRequest object.

  Fields:
    azureClientId: Required. A client provided ID the resource. Must be unique
      within the parent resource. The provided ID will be part of the
      AzureClient resource name formatted as
      `projects//locations//azureClients/`. Valid characters are `/a-z-/`.
      Cannot be longer than 63 characters.
    googleCloudGkemulticloudV1AzureClient: A
      GoogleCloudGkemulticloudV1AzureClient resource to be passed as the
      request body.
    parent: Required. The parent location where this AzureClient resource will
      be created. Location names are formatted as `projects//locations/`. See
      [Resource Names](https://cloud.google.com/apis/design/resource_names)
      for more details on Google Cloud resource names.
    validateOnly: If set, only validate the request, but do not actually
      create the client.
  """
    azureClientId = _messages.StringField(1)
    googleCloudGkemulticloudV1AzureClient = _messages.MessageField('GoogleCloudGkemulticloudV1AzureClient', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)