from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsApisRevisionsDebugsessionsDataService(base_api.BaseApiService):
    """Service class for the organizations_environments_apis_revisions_debugsessions_data resource."""
    _NAME = 'organizations_environments_apis_revisions_debugsessions_data'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsApisRevisionsDebugsessionsDataService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the debug data from a transaction.

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsDataGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DebugSessionTransaction) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apis/{apisId}/revisions/{revisionsId}/debugsessions/{debugsessionsId}/data/{dataId}', http_method='GET', method_id='apigee.organizations.environments.apis.revisions.debugsessions.data.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsDataGetRequest', response_type_name='GoogleCloudApigeeV1DebugSessionTransaction', supports_download=False)