from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
class AppsAuthorizedCertificatesService(base_api.BaseApiService):
    """Service class for the apps_authorizedCertificates resource."""
    _NAME = 'apps_authorizedCertificates'

    def __init__(self, client):
        super(AppengineV1beta.AppsAuthorizedCertificatesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Uploads the specified SSL certificate.

      Args:
        request: (AppengineAppsAuthorizedCertificatesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuthorizedCertificate) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/authorizedCertificates', http_method='POST', method_id='appengine.apps.authorizedCertificates.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta/{+parent}/authorizedCertificates', request_field='authorizedCertificate', request_type_name='AppengineAppsAuthorizedCertificatesCreateRequest', response_type_name='AuthorizedCertificate', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified SSL certificate.

      Args:
        request: (AppengineAppsAuthorizedCertificatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/authorizedCertificates/{authorizedCertificatesId}', http_method='DELETE', method_id='appengine.apps.authorizedCertificates.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsAuthorizedCertificatesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified SSL certificate.

      Args:
        request: (AppengineAppsAuthorizedCertificatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuthorizedCertificate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/authorizedCertificates/{authorizedCertificatesId}', http_method='GET', method_id='appengine.apps.authorizedCertificates.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsAuthorizedCertificatesGetRequest', response_type_name='AuthorizedCertificate', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all SSL certificates the user is authorized to administer.

      Args:
        request: (AppengineAppsAuthorizedCertificatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAuthorizedCertificatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/authorizedCertificates', http_method='GET', method_id='appengine.apps.authorizedCertificates.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v1beta/{+parent}/authorizedCertificates', request_field='', request_type_name='AppengineAppsAuthorizedCertificatesListRequest', response_type_name='ListAuthorizedCertificatesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified SSL certificate. To renew a certificate and maintain its existing domain mappings, update certificate_data with a new certificate. The new certificate must be applicable to the same domains as the original certificate. The certificate display_name may also be updated.

      Args:
        request: (AppengineAppsAuthorizedCertificatesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuthorizedCertificate) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/authorizedCertificates/{authorizedCertificatesId}', http_method='PATCH', method_id='appengine.apps.authorizedCertificates.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='authorizedCertificate', request_type_name='AppengineAppsAuthorizedCertificatesPatchRequest', response_type_name='AuthorizedCertificate', supports_download=False)