from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudquotas.v1 import cloudquotas_v1_messages as messages
class OrganizationsLocationsServicesService(base_api.BaseApiService):
    """Service class for the organizations_locations_services resource."""
    _NAME = 'organizations_locations_services'

    def __init__(self, client):
        super(CloudquotasV1.OrganizationsLocationsServicesService, self).__init__(client)
        self._upload_configs = {}