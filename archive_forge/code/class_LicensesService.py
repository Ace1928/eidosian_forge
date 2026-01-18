from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class LicensesService(base_api.BaseApiService):
    """Service class for the licenses resource."""
    _NAME = 'licenses'

    def __init__(self, client):
        super(ComputeBeta.LicensesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified license. *Caution* This resource is intended for use only by third-party partners who are creating Cloud Marketplace images. .

      Args:
        request: (ComputeLicensesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.licenses.delete', ordered_params=['project', 'license'], path_params=['license', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/licenses/{license}', request_field='', request_type_name='ComputeLicensesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified License resource. *Caution* This resource is intended for use only by third-party partners who are creating Cloud Marketplace images. .

      Args:
        request: (ComputeLicensesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (License) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.licenses.get', ordered_params=['project', 'license'], path_params=['license', 'project'], query_params=[], relative_path='projects/{project}/global/licenses/{license}', request_field='', request_type_name='ComputeLicensesGetRequest', response_type_name='License', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists. *Caution* This resource is intended for use only by third-party partners who are creating Cloud Marketplace images. .

      Args:
        request: (ComputeLicensesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.licenses.getIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/global/licenses/{resource}/getIamPolicy', request_field='', request_type_name='ComputeLicensesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Create a License resource in the specified project. *Caution* This resource is intended for use only by third-party partners who are creating Cloud Marketplace images. .

      Args:
        request: (ComputeLicensesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.licenses.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/licenses', request_field='license', request_type_name='ComputeLicensesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of licenses available in the specified project. This method does not get any licenses that belong to other projects, including licenses attached to publicly-available images, like Debian 9. If you want to get a list of publicly-available licenses, use this method to make a request to the respective image project, such as debian-cloud or windows-cloud. *Caution* This resource is intended for use only by third-party partners who are creating Cloud Marketplace images. .

      Args:
        request: (ComputeLicensesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LicensesListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.licenses.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/licenses', request_field='', request_type_name='ComputeLicensesListRequest', response_type_name='LicensesListResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. *Caution* This resource is intended for use only by third-party partners who are creating Cloud Marketplace images. .

      Args:
        request: (ComputeLicensesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.licenses.setIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/licenses/{resource}/setIamPolicy', request_field='globalSetPolicyRequest', request_type_name='ComputeLicensesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. *Caution* This resource is intended for use only by third-party partners who are creating Cloud Marketplace images. .

      Args:
        request: (ComputeLicensesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.licenses.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/licenses/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeLicensesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)