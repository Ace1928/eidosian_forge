from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsInstancesAttachmentsService(base_api.BaseApiService):
    """Service class for the organizations_instances_attachments resource."""
    _NAME = 'organizations_instances_attachments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsInstancesAttachmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new attachment of an environment to an instance. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesAttachmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/attachments', http_method='POST', method_id='apigee.organizations.instances.attachments.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/attachments', request_field='googleCloudApigeeV1InstanceAttachment', request_type_name='ApigeeOrganizationsInstancesAttachmentsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an attachment. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesAttachmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/attachments/{attachmentsId}', http_method='DELETE', method_id='apigee.organizations.instances.attachments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsInstancesAttachmentsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an attachment. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesAttachmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1InstanceAttachment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/attachments/{attachmentsId}', http_method='GET', method_id='apigee.organizations.instances.attachments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsInstancesAttachmentsGetRequest', response_type_name='GoogleCloudApigeeV1InstanceAttachment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all attachments to an instance. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesAttachmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListInstanceAttachmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/attachments', http_method='GET', method_id='apigee.organizations.instances.attachments.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/attachments', request_field='', request_type_name='ApigeeOrganizationsInstancesAttachmentsListRequest', response_type_name='GoogleCloudApigeeV1ListInstanceAttachmentsResponse', supports_download=False)