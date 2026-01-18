from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvgroupsAttachmentsService(base_api.BaseApiService):
    """Service class for the organizations_envgroups_attachments resource."""
    _NAME = 'organizations_envgroups_attachments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvgroupsAttachmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new attachment of an environment to an environment group.

      Args:
        request: (ApigeeOrganizationsEnvgroupsAttachmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/envgroups/{envgroupsId}/attachments', http_method='POST', method_id='apigee.organizations.envgroups.attachments.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/attachments', request_field='googleCloudApigeeV1EnvironmentGroupAttachment', request_type_name='ApigeeOrganizationsEnvgroupsAttachmentsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an environment group attachment.

      Args:
        request: (ApigeeOrganizationsEnvgroupsAttachmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/envgroups/{envgroupsId}/attachments/{attachmentsId}', http_method='DELETE', method_id='apigee.organizations.envgroups.attachments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvgroupsAttachmentsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an environment group attachment.

      Args:
        request: (ApigeeOrganizationsEnvgroupsAttachmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1EnvironmentGroupAttachment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/envgroups/{envgroupsId}/attachments/{attachmentsId}', http_method='GET', method_id='apigee.organizations.envgroups.attachments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvgroupsAttachmentsGetRequest', response_type_name='GoogleCloudApigeeV1EnvironmentGroupAttachment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all attachments of an environment group.

      Args:
        request: (ApigeeOrganizationsEnvgroupsAttachmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListEnvironmentGroupAttachmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/envgroups/{envgroupsId}/attachments', http_method='GET', method_id='apigee.organizations.envgroups.attachments.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/attachments', request_field='', request_type_name='ApigeeOrganizationsEnvgroupsAttachmentsListRequest', response_type_name='GoogleCloudApigeeV1ListEnvironmentGroupAttachmentsResponse', supports_download=False)