from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class OrganizationsLocationsStandardsService(base_api.BaseApiService):
    """Service class for the organizations_locations_standards resource."""
    _NAME = 'organizations_locations_standards'

    def __init__(self, client):
        super(AuditmanagerV1alpha.OrganizationsLocationsStandardsService, self).__init__(client)
        self._upload_configs = {}