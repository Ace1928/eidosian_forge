from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsInstancesService(base_api.BaseApiService):
    """Service class for the organizations_instances resource."""
    _NAME = 'organizations_instances'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsInstancesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an Apigee runtime instance. The instance is accessible from the authorized network configured on the organization. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances', http_method='POST', method_id='apigee.organizations.instances.create', ordered_params=['parent'], path_params=['parent'], query_params=['environments', 'runtimeVersion'], relative_path='v1/{+parent}/instances', request_field='googleCloudApigeeV1Instance', request_type_name='ApigeeOrganizationsInstancesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an Apigee runtime instance. The instance stops serving requests and the runtime data is deleted. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}', http_method='DELETE', method_id='apigee.organizations.instances.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsInstancesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details for an Apigee runtime instance. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Instance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}', http_method='GET', method_id='apigee.organizations.instances.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsInstancesGetRequest', response_type_name='GoogleCloudApigeeV1Instance', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all Apigee runtime instances for the organization. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListInstancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances', http_method='GET', method_id='apigee.organizations.instances.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/instances', request_field='', request_type_name='ApigeeOrganizationsInstancesListRequest', response_type_name='GoogleCloudApigeeV1ListInstancesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an Apigee runtime instance. You can update the fields described in NodeConfig. No other fields will be updated. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}', http_method='PATCH', method_id='apigee.organizations.instances.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1Instance', request_type_name='ApigeeOrganizationsInstancesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def ReportStatus(self, request, global_params=None):
        """Reports the latest status for a runtime instance.

      Args:
        request: (ApigeeOrganizationsInstancesReportStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ReportInstanceStatusResponse) The response message.
      """
        config = self.GetMethodConfig('ReportStatus')
        return self._RunMethod(config, request, global_params=global_params)
    ReportStatus.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}:reportStatus', http_method='POST', method_id='apigee.organizations.instances.reportStatus', ordered_params=['instance'], path_params=['instance'], query_params=[], relative_path='v1/{+instance}:reportStatus', request_field='googleCloudApigeeV1ReportInstanceStatusRequest', request_type_name='ApigeeOrganizationsInstancesReportStatusRequest', response_type_name='GoogleCloudApigeeV1ReportInstanceStatusResponse', supports_download=False)