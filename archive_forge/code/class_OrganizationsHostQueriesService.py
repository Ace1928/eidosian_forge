from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsHostQueriesService(base_api.BaseApiService):
    """Service class for the organizations_hostQueries resource."""
    _NAME = 'organizations_hostQueries'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsHostQueriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Submit a query at host level to be processed in the background. If the submission of the query succeeds, the API returns a 201 status and an ID that refer to the query. In addition to the HTTP status 201, the `state` of "enqueued" means that the request succeeded.

      Args:
        request: (ApigeeOrganizationsHostQueriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1AsyncQuery) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/hostQueries', http_method='POST', method_id='apigee.organizations.hostQueries.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/hostQueries', request_field='googleCloudApigeeV1Query', request_type_name='ApigeeOrganizationsHostQueriesCreateRequest', response_type_name='GoogleCloudApigeeV1AsyncQuery', supports_download=False)

    def Get(self, request, global_params=None):
        """Get status of a query submitted at host level. If the query is still in progress, the `state` is set to "running" After the query has completed successfully, `state` is set to "completed".

      Args:
        request: (ApigeeOrganizationsHostQueriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1AsyncQuery) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/hostQueries/{hostQueriesId}', http_method='GET', method_id='apigee.organizations.hostQueries.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsHostQueriesGetRequest', response_type_name='GoogleCloudApigeeV1AsyncQuery', supports_download=False)

    def GetResult(self, request, global_params=None):
        """After the query is completed, use this API to retrieve the results. If the request succeeds, and there is a non-zero result set, the result is downloaded to the client as a zipped JSON file. The name of the downloaded file will be: OfflineQueryResult-.zip Example: `OfflineQueryResult-9cfc0d85-0f30-46d6-ae6f-318d0cb961bd.zip`.

      Args:
        request: (ApigeeOrganizationsHostQueriesGetResultRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('GetResult')
        return self._RunMethod(config, request, global_params=global_params)
    GetResult.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/hostQueries/{hostQueriesId}/result', http_method='GET', method_id='apigee.organizations.hostQueries.getResult', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsHostQueriesGetResultRequest', response_type_name='GoogleApiHttpBody', supports_download=False)

    def GetResultView(self, request, global_params=None):
        """GetResultView method for the organizations_hostQueries service.

      Args:
        request: (ApigeeOrganizationsHostQueriesGetResultViewRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1AsyncQueryResultView) The response message.
      """
        config = self.GetMethodConfig('GetResultView')
        return self._RunMethod(config, request, global_params=global_params)
    GetResultView.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/hostQueries/{hostQueriesId}/resultView', http_method='GET', method_id='apigee.organizations.hostQueries.getResultView', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsHostQueriesGetResultViewRequest', response_type_name='GoogleCloudApigeeV1AsyncQueryResultView', supports_download=False)

    def List(self, request, global_params=None):
        """Return a list of Asynchronous Queries at host level.

      Args:
        request: (ApigeeOrganizationsHostQueriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListAsyncQueriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/hostQueries', http_method='GET', method_id='apigee.organizations.hostQueries.list', ordered_params=['parent'], path_params=['parent'], query_params=['dataset', 'envgroupHostname', 'from_', 'inclQueriesWithoutReport', 'status', 'submittedBy', 'to'], relative_path='v1/{+parent}/hostQueries', request_field='', request_type_name='ApigeeOrganizationsHostQueriesListRequest', response_type_name='GoogleCloudApigeeV1ListAsyncQueriesResponse', supports_download=False)