from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClientsGetRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAzureClientsGetRequest object.

  Fields:
    name: Required. The name of the AzureClient resource to describe.
      AzureClient names are formatted as `projects//locations//azureClients/`.
      See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
  """
    name = _messages.StringField(1, required=True)