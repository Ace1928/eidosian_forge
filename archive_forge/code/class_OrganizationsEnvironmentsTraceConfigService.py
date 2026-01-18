from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsTraceConfigService(base_api.BaseApiService):
    """Service class for the organizations_environments_traceConfig resource."""
    _NAME = 'organizations_environments_traceConfig'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsTraceConfigService, self).__init__(client)
        self._upload_configs = {}