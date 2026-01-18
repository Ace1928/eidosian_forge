from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
class OrganizationsLocationsGlobalService(base_api.BaseApiService):
    """Service class for the organizations_locations_global resource."""
    _NAME = 'organizations_locations_global'

    def __init__(self, client):
        super(BeyondcorpV1alpha.OrganizationsLocationsGlobalService, self).__init__(client)
        self._upload_configs = {}