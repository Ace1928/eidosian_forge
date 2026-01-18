from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEndpointAttachmentsService(base_api.BaseApiService):
    """Service class for the organizations_endpointAttachments resource."""
    _NAME = 'organizations_endpointAttachments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEndpointAttachmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an endpoint attachment. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsEndpointAttachmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/endpointAttachments', http_method='POST', method_id='apigee.organizations.endpointAttachments.create', ordered_params=['parent'], path_params=['parent'], query_params=['endpointAttachmentId'], relative_path='v1/{+parent}/endpointAttachments', request_field='googleCloudApigeeV1EndpointAttachment', request_type_name='ApigeeOrganizationsEndpointAttachmentsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an endpoint attachment.

      Args:
        request: (ApigeeOrganizationsEndpointAttachmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/endpointAttachments/{endpointAttachmentsId}', http_method='DELETE', method_id='apigee.organizations.endpointAttachments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEndpointAttachmentsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the endpoint attachment.

      Args:
        request: (ApigeeOrganizationsEndpointAttachmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1EndpointAttachment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/endpointAttachments/{endpointAttachmentsId}', http_method='GET', method_id='apigee.organizations.endpointAttachments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEndpointAttachmentsGetRequest', response_type_name='GoogleCloudApigeeV1EndpointAttachment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the endpoint attachments in an organization.

      Args:
        request: (ApigeeOrganizationsEndpointAttachmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListEndpointAttachmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/endpointAttachments', http_method='GET', method_id='apigee.organizations.endpointAttachments.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/endpointAttachments', request_field='', request_type_name='ApigeeOrganizationsEndpointAttachmentsListRequest', response_type_name='GoogleCloudApigeeV1ListEndpointAttachmentsResponse', supports_download=False)