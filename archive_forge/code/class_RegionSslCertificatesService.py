from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionSslCertificatesService(base_api.BaseApiService):
    """Service class for the regionSslCertificates resource."""
    _NAME = 'regionSslCertificates'

    def __init__(self, client):
        super(ComputeBeta.RegionSslCertificatesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified SslCertificate resource in the region.

      Args:
        request: (ComputeRegionSslCertificatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionSslCertificates.delete', ordered_params=['project', 'region', 'sslCertificate'], path_params=['project', 'region', 'sslCertificate'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/sslCertificates/{sslCertificate}', request_field='', request_type_name='ComputeRegionSslCertificatesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified SslCertificate resource in the specified region. Get a list of available SSL certificates by making a list() request.

      Args:
        request: (ComputeRegionSslCertificatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslCertificate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionSslCertificates.get', ordered_params=['project', 'region', 'sslCertificate'], path_params=['project', 'region', 'sslCertificate'], query_params=[], relative_path='projects/{project}/regions/{region}/sslCertificates/{sslCertificate}', request_field='', request_type_name='ComputeRegionSslCertificatesGetRequest', response_type_name='SslCertificate', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a SslCertificate resource in the specified project and region using the data included in the request.

      Args:
        request: (ComputeRegionSslCertificatesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionSslCertificates.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/sslCertificates', request_field='sslCertificate', request_type_name='ComputeRegionSslCertificatesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of SslCertificate resources available to the specified project in the specified region.

      Args:
        request: (ComputeRegionSslCertificatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslCertificateList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionSslCertificates.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/sslCertificates', request_field='', request_type_name='ComputeRegionSslCertificatesListRequest', response_type_name='SslCertificateList', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource and region.

      Args:
        request: (ComputeRegionSslCertificatesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionSslCertificates.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/sslCertificates/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionSslCertificatesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)