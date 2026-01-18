from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsApisKeyvaluemapsService(base_api.BaseApiService):
    """Service class for the organizations_apis_keyvaluemaps resource."""
    _NAME = 'organizations_apis_keyvaluemaps'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsApisKeyvaluemapsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a key value map in an API proxy.

      Args:
        request: (ApigeeOrganizationsApisKeyvaluemapsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1KeyValueMap) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apis/{apisId}/keyvaluemaps', http_method='POST', method_id='apigee.organizations.apis.keyvaluemaps.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/keyvaluemaps', request_field='googleCloudApigeeV1KeyValueMap', request_type_name='ApigeeOrganizationsApisKeyvaluemapsCreateRequest', response_type_name='GoogleCloudApigeeV1KeyValueMap', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a key value map from an API proxy.

      Args:
        request: (ApigeeOrganizationsApisKeyvaluemapsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1KeyValueMap) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apis/{apisId}/keyvaluemaps/{keyvaluemapsId}', http_method='DELETE', method_id='apigee.organizations.apis.keyvaluemaps.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApisKeyvaluemapsDeleteRequest', response_type_name='GoogleCloudApigeeV1KeyValueMap', supports_download=False)