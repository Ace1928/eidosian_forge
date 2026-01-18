from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsAnalyticsDatastoresService(base_api.BaseApiService):
    """Service class for the organizations_analytics_datastores resource."""
    _NAME = 'organizations_analytics_datastores'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsAnalyticsDatastoresService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a Datastore for an org.

      Args:
        request: (ApigeeOrganizationsAnalyticsDatastoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Datastore) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/analytics/datastores', http_method='POST', method_id='apigee.organizations.analytics.datastores.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/analytics/datastores', request_field='googleCloudApigeeV1Datastore', request_type_name='ApigeeOrganizationsAnalyticsDatastoresCreateRequest', response_type_name='GoogleCloudApigeeV1Datastore', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a Datastore from an org.

      Args:
        request: (ApigeeOrganizationsAnalyticsDatastoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/analytics/datastores/{datastoresId}', http_method='DELETE', method_id='apigee.organizations.analytics.datastores.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsAnalyticsDatastoresDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get a Datastore.

      Args:
        request: (ApigeeOrganizationsAnalyticsDatastoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Datastore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/analytics/datastores/{datastoresId}', http_method='GET', method_id='apigee.organizations.analytics.datastores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsAnalyticsDatastoresGetRequest', response_type_name='GoogleCloudApigeeV1Datastore', supports_download=False)

    def List(self, request, global_params=None):
        """List Datastores.

      Args:
        request: (ApigeeOrganizationsAnalyticsDatastoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListDatastoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/analytics/datastores', http_method='GET', method_id='apigee.organizations.analytics.datastores.list', ordered_params=['parent'], path_params=['parent'], query_params=['targetType'], relative_path='v1/{+parent}/analytics/datastores', request_field='', request_type_name='ApigeeOrganizationsAnalyticsDatastoresListRequest', response_type_name='GoogleCloudApigeeV1ListDatastoresResponse', supports_download=False)

    def Test(self, request, global_params=None):
        """Test if Datastore configuration is correct. This includes checking if credentials provided by customer have required permissions in target destination storage.

      Args:
        request: (ApigeeOrganizationsAnalyticsDatastoresTestRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TestDatastoreResponse) The response message.
      """
        config = self.GetMethodConfig('Test')
        return self._RunMethod(config, request, global_params=global_params)
    Test.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/analytics/datastores:test', http_method='POST', method_id='apigee.organizations.analytics.datastores.test', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/analytics/datastores:test', request_field='googleCloudApigeeV1Datastore', request_type_name='ApigeeOrganizationsAnalyticsDatastoresTestRequest', response_type_name='GoogleCloudApigeeV1TestDatastoreResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Update a Datastore.

      Args:
        request: (ApigeeOrganizationsAnalyticsDatastoresUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Datastore) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/analytics/datastores/{datastoresId}', http_method='PUT', method_id='apigee.organizations.analytics.datastores.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='googleCloudApigeeV1Datastore', request_type_name='ApigeeOrganizationsAnalyticsDatastoresUpdateRequest', response_type_name='GoogleCloudApigeeV1Datastore', supports_download=False)