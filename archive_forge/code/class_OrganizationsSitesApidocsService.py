from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsSitesApidocsService(base_api.BaseApiService):
    """Service class for the organizations_sites_apidocs resource."""
    _NAME = 'organizations_sites_apidocs'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsSitesApidocsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new catalog item.

      Args:
        request: (ApigeeOrganizationsSitesApidocsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiDocResponse) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apidocs', http_method='POST', method_id='apigee.organizations.sites.apidocs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/apidocs', request_field='googleCloudApigeeV1ApiDoc', request_type_name='ApigeeOrganizationsSitesApidocsCreateRequest', response_type_name='GoogleCloudApigeeV1ApiDocResponse', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a catalog item.

      Args:
        request: (ApigeeOrganizationsSitesApidocsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apidocs/{apidocsId}', http_method='DELETE', method_id='apigee.organizations.sites.apidocs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSitesApidocsDeleteRequest', response_type_name='GoogleCloudApigeeV1DeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a catalog item.

      Args:
        request: (ApigeeOrganizationsSitesApidocsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiDocResponse) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apidocs/{apidocsId}', http_method='GET', method_id='apigee.organizations.sites.apidocs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSitesApidocsGetRequest', response_type_name='GoogleCloudApigeeV1ApiDocResponse', supports_download=False)

    def GetDocumentation(self, request, global_params=None):
        """Gets the documentation for the specified catalog item.

      Args:
        request: (ApigeeOrganizationsSitesApidocsGetDocumentationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiDocDocumentationResponse) The response message.
      """
        config = self.GetMethodConfig('GetDocumentation')
        return self._RunMethod(config, request, global_params=global_params)
    GetDocumentation.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apidocs/{apidocsId}/documentation', http_method='GET', method_id='apigee.organizations.sites.apidocs.getDocumentation', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSitesApidocsGetDocumentationRequest', response_type_name='GoogleCloudApigeeV1ApiDocDocumentationResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the catalog items associated with a portal.

      Args:
        request: (ApigeeOrganizationsSitesApidocsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListApiDocsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apidocs', http_method='GET', method_id='apigee.organizations.sites.apidocs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/apidocs', request_field='', request_type_name='ApigeeOrganizationsSitesApidocsListRequest', response_type_name='GoogleCloudApigeeV1ListApiDocsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a catalog item.

      Args:
        request: (ApigeeOrganizationsSitesApidocsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiDocResponse) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apidocs/{apidocsId}', http_method='PUT', method_id='apigee.organizations.sites.apidocs.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='googleCloudApigeeV1ApiDoc', request_type_name='ApigeeOrganizationsSitesApidocsUpdateRequest', response_type_name='GoogleCloudApigeeV1ApiDocResponse', supports_download=False)

    def UpdateDocumentation(self, request, global_params=None):
        """Updates the documentation for the specified catalog item. Note that the documentation file contents will not be populated in the return message.

      Args:
        request: (ApigeeOrganizationsSitesApidocsUpdateDocumentationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiDocDocumentationResponse) The response message.
      """
        config = self.GetMethodConfig('UpdateDocumentation')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateDocumentation.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sites/{sitesId}/apidocs/{apidocsId}/documentation', http_method='PATCH', method_id='apigee.organizations.sites.apidocs.updateDocumentation', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='googleCloudApigeeV1ApiDocDocumentation', request_type_name='ApigeeOrganizationsSitesApidocsUpdateDocumentationRequest', response_type_name='GoogleCloudApigeeV1ApiDocDocumentationResponse', supports_download=False)