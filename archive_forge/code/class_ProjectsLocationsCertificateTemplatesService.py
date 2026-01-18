from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privateca.v1 import privateca_v1_messages as messages
class ProjectsLocationsCertificateTemplatesService(base_api.BaseApiService):
    """Service class for the projects_locations_certificateTemplates resource."""
    _NAME = 'projects_locations_certificateTemplates'

    def __init__(self, client):
        super(PrivatecaV1.ProjectsLocationsCertificateTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new CertificateTemplate in a given Project and Location.

      Args:
        request: (PrivatecaProjectsLocationsCertificateTemplatesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateTemplates', http_method='POST', method_id='privateca.projects.locations.certificateTemplates.create', ordered_params=['parent'], path_params=['parent'], query_params=['certificateTemplateId', 'requestId'], relative_path='v1/{+parent}/certificateTemplates', request_field='certificateTemplate', request_type_name='PrivatecaProjectsLocationsCertificateTemplatesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """DeleteCertificateTemplate deletes a CertificateTemplate.

      Args:
        request: (PrivatecaProjectsLocationsCertificateTemplatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateTemplates/{certificateTemplatesId}', http_method='DELETE', method_id='privateca.projects.locations.certificateTemplates.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='PrivatecaProjectsLocationsCertificateTemplatesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a CertificateTemplate.

      Args:
        request: (PrivatecaProjectsLocationsCertificateTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CertificateTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateTemplates/{certificateTemplatesId}', http_method='GET', method_id='privateca.projects.locations.certificateTemplates.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='PrivatecaProjectsLocationsCertificateTemplatesGetRequest', response_type_name='CertificateTemplate', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (PrivatecaProjectsLocationsCertificateTemplatesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateTemplates/{certificateTemplatesId}:getIamPolicy', http_method='GET', method_id='privateca.projects.locations.certificateTemplates.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='PrivatecaProjectsLocationsCertificateTemplatesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CertificateTemplates.

      Args:
        request: (PrivatecaProjectsLocationsCertificateTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCertificateTemplatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateTemplates', http_method='GET', method_id='privateca.projects.locations.certificateTemplates.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/certificateTemplates', request_field='', request_type_name='PrivatecaProjectsLocationsCertificateTemplatesListRequest', response_type_name='ListCertificateTemplatesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a CertificateTemplate.

      Args:
        request: (PrivatecaProjectsLocationsCertificateTemplatesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateTemplates/{certificateTemplatesId}', http_method='PATCH', method_id='privateca.projects.locations.certificateTemplates.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='certificateTemplate', request_type_name='PrivatecaProjectsLocationsCertificateTemplatesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (PrivatecaProjectsLocationsCertificateTemplatesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateTemplates/{certificateTemplatesId}:setIamPolicy', http_method='POST', method_id='privateca.projects.locations.certificateTemplates.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='PrivatecaProjectsLocationsCertificateTemplatesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (PrivatecaProjectsLocationsCertificateTemplatesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/certificateTemplates/{certificateTemplatesId}:testIamPermissions', http_method='POST', method_id='privateca.projects.locations.certificateTemplates.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='PrivatecaProjectsLocationsCertificateTemplatesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)