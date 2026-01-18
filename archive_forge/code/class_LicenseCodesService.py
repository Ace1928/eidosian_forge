from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class LicenseCodesService(base_api.BaseApiService):
    """Service class for the licenseCodes resource."""
    _NAME = 'licenseCodes'

    def __init__(self, client):
        super(ComputeBeta.LicenseCodesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Return a specified license code. License codes are mirrored across all projects that have permissions to read the License Code. *Caution* This resource is intended for use only by third-party partners who are creating Cloud Marketplace images. .

      Args:
        request: (ComputeLicenseCodesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LicenseCode) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.licenseCodes.get', ordered_params=['project', 'licenseCode'], path_params=['licenseCode', 'project'], query_params=[], relative_path='projects/{project}/global/licenseCodes/{licenseCode}', request_field='', request_type_name='ComputeLicenseCodesGetRequest', response_type_name='LicenseCode', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. *Caution* This resource is intended for use only by third-party partners who are creating Cloud Marketplace images. .

      Args:
        request: (ComputeLicenseCodesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.licenseCodes.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/licenseCodes/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeLicenseCodesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)