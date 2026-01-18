from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class OrganizationsInspectTemplatesService(base_api.BaseApiService):
    """Service class for the organizations_inspectTemplates resource."""
    _NAME = 'organizations_inspectTemplates'

    def __init__(self, client):
        super(DlpV2.OrganizationsInspectTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an InspectTemplate for reusing frequently used configuration for inspecting content, images, and storage. See https://cloud.google.com/sensitive-data-protection/docs/creating-templates to learn more.

      Args:
        request: (DlpOrganizationsInspectTemplatesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2InspectTemplate) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/inspectTemplates', http_method='POST', method_id='dlp.organizations.inspectTemplates.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/inspectTemplates', request_field='googlePrivacyDlpV2CreateInspectTemplateRequest', request_type_name='DlpOrganizationsInspectTemplatesCreateRequest', response_type_name='GooglePrivacyDlpV2InspectTemplate', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an InspectTemplate. See https://cloud.google.com/sensitive-data-protection/docs/creating-templates to learn more.

      Args:
        request: (DlpOrganizationsInspectTemplatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/inspectTemplates/{inspectTemplatesId}', http_method='DELETE', method_id='dlp.organizations.inspectTemplates.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpOrganizationsInspectTemplatesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an InspectTemplate. See https://cloud.google.com/sensitive-data-protection/docs/creating-templates to learn more.

      Args:
        request: (DlpOrganizationsInspectTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2InspectTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/inspectTemplates/{inspectTemplatesId}', http_method='GET', method_id='dlp.organizations.inspectTemplates.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpOrganizationsInspectTemplatesGetRequest', response_type_name='GooglePrivacyDlpV2InspectTemplate', supports_download=False)

    def List(self, request, global_params=None):
        """Lists InspectTemplates. See https://cloud.google.com/sensitive-data-protection/docs/creating-templates to learn more.

      Args:
        request: (DlpOrganizationsInspectTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListInspectTemplatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/inspectTemplates', http_method='GET', method_id='dlp.organizations.inspectTemplates.list', ordered_params=['parent'], path_params=['parent'], query_params=['locationId', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/inspectTemplates', request_field='', request_type_name='DlpOrganizationsInspectTemplatesListRequest', response_type_name='GooglePrivacyDlpV2ListInspectTemplatesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the InspectTemplate. See https://cloud.google.com/sensitive-data-protection/docs/creating-templates to learn more.

      Args:
        request: (DlpOrganizationsInspectTemplatesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2InspectTemplate) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/inspectTemplates/{inspectTemplatesId}', http_method='PATCH', method_id='dlp.organizations.inspectTemplates.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='googlePrivacyDlpV2UpdateInspectTemplateRequest', request_type_name='DlpOrganizationsInspectTemplatesPatchRequest', response_type_name='GooglePrivacyDlpV2InspectTemplate', supports_download=False)