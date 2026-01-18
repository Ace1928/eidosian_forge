from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1alpha2 import binaryauthorization_v1alpha2_messages as messages
class ProjectsAttestorsAttestationsService(base_api.BaseApiService):
    """Service class for the projects_attestors_attestations resource."""
    _NAME = 'projects_attestors_attestations'

    def __init__(self, client):
        super(BinaryauthorizationV1alpha2.ProjectsAttestorsAttestationsService, self).__init__(client)
        self._upload_configs = {}

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (BinaryauthorizationProjectsAttestorsAttestationsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/attestors/{attestorsId}/attestations/{attestationsId}:testIamPermissions', http_method='POST', method_id='binaryauthorization.projects.attestors.attestations.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='BinaryauthorizationProjectsAttestorsAttestationsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)