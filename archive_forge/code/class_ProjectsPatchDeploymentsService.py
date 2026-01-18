from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1beta import osconfig_v1beta_messages as messages
class ProjectsPatchDeploymentsService(base_api.BaseApiService):
    """Service class for the projects_patchDeployments resource."""
    _NAME = 'projects_patchDeployments'

    def __init__(self, client):
        super(OsconfigV1beta.ProjectsPatchDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an OS Config patch deployment.

      Args:
        request: (OsconfigProjectsPatchDeploymentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PatchDeployment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchDeployments', http_method='POST', method_id='osconfig.projects.patchDeployments.create', ordered_params=['parent'], path_params=['parent'], query_params=['patchDeploymentId'], relative_path='v1beta/{+parent}/patchDeployments', request_field='patchDeployment', request_type_name='OsconfigProjectsPatchDeploymentsCreateRequest', response_type_name='PatchDeployment', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete an OS Config patch deployment.

      Args:
        request: (OsconfigProjectsPatchDeploymentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchDeployments/{patchDeploymentsId}', http_method='DELETE', method_id='osconfig.projects.patchDeployments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='OsconfigProjectsPatchDeploymentsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get an OS Config patch deployment.

      Args:
        request: (OsconfigProjectsPatchDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PatchDeployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchDeployments/{patchDeploymentsId}', http_method='GET', method_id='osconfig.projects.patchDeployments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='OsconfigProjectsPatchDeploymentsGetRequest', response_type_name='PatchDeployment', supports_download=False)

    def List(self, request, global_params=None):
        """Get a page of OS Config patch deployments.

      Args:
        request: (OsconfigProjectsPatchDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPatchDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchDeployments', http_method='GET', method_id='osconfig.projects.patchDeployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/patchDeployments', request_field='', request_type_name='OsconfigProjectsPatchDeploymentsListRequest', response_type_name='ListPatchDeploymentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update an OS Config patch deployment.

      Args:
        request: (OsconfigProjectsPatchDeploymentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PatchDeployment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchDeployments/{patchDeploymentsId}', http_method='PATCH', method_id='osconfig.projects.patchDeployments.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='patchDeployment', request_type_name='OsconfigProjectsPatchDeploymentsPatchRequest', response_type_name='PatchDeployment', supports_download=False)

    def Pause(self, request, global_params=None):
        """Change state of patch deployment to "PAUSED". Patch deployment in paused state doesn't generate patch jobs.

      Args:
        request: (OsconfigProjectsPatchDeploymentsPauseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PatchDeployment) The response message.
      """
        config = self.GetMethodConfig('Pause')
        return self._RunMethod(config, request, global_params=global_params)
    Pause.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchDeployments/{patchDeploymentsId}:pause', http_method='POST', method_id='osconfig.projects.patchDeployments.pause', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:pause', request_field='pausePatchDeploymentRequest', request_type_name='OsconfigProjectsPatchDeploymentsPauseRequest', response_type_name='PatchDeployment', supports_download=False)

    def Resume(self, request, global_params=None):
        """Change state of patch deployment back to "ACTIVE". Patch deployment in active state continues to generate patch jobs.

      Args:
        request: (OsconfigProjectsPatchDeploymentsResumeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PatchDeployment) The response message.
      """
        config = self.GetMethodConfig('Resume')
        return self._RunMethod(config, request, global_params=global_params)
    Resume.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchDeployments/{patchDeploymentsId}:resume', http_method='POST', method_id='osconfig.projects.patchDeployments.resume', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:resume', request_field='resumePatchDeploymentRequest', request_type_name='OsconfigProjectsPatchDeploymentsResumeRequest', response_type_name='PatchDeployment', supports_download=False)