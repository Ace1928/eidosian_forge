from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.azure import resource_args
from googlecloudsdk.command_lib.container.gkemulticloud import flags
class ClientsClient(_AzureClientBase):
    """Client for Azure Clients in the gkemulticloud API."""

    def __init__(self, **kwargs):
        super(ClientsClient, self).__init__(**kwargs)
        self._service = self._client.projects_locations_azureClients
        self._list_result_field = 'azureClients'

    def Create(self, client_ref, args):
        """Creates a new Azure client."""
        req = self._messages.GkemulticloudProjectsLocationsAzureClientsCreateRequest(googleCloudGkemulticloudV1AzureClient=self._Client(client_ref, args), azureClientId=client_ref.azureClientsId, parent=client_ref.Parent().RelativeName(), validateOnly=flags.GetValidateOnly(args))
        return self._service.Create(req)