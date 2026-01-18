from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class MachineImagesService(base_api.BaseApiService):
    """Service class for the machineImages resource."""
    _NAME = 'machineImages'

    def __init__(self, client):
        super(ComputeBeta.MachineImagesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified machine image. Deleting a machine image is permanent and cannot be undone.

      Args:
        request: (ComputeMachineImagesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.machineImages.delete', ordered_params=['project', 'machineImage'], path_params=['machineImage', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/machineImages/{machineImage}', request_field='', request_type_name='ComputeMachineImagesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified machine image.

      Args:
        request: (ComputeMachineImagesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MachineImage) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.machineImages.get', ordered_params=['project', 'machineImage'], path_params=['machineImage', 'project'], query_params=[], relative_path='projects/{project}/global/machineImages/{machineImage}', request_field='', request_type_name='ComputeMachineImagesGetRequest', response_type_name='MachineImage', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeMachineImagesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.machineImages.getIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/global/machineImages/{resource}/getIamPolicy', request_field='', request_type_name='ComputeMachineImagesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a machine image in the specified project using the data that is included in the request. If you are creating a new machine image to update an existing instance, your new machine image should use the same network or, if applicable, the same subnetwork as the original instance.

      Args:
        request: (ComputeMachineImagesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.machineImages.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId', 'sourceInstance'], relative_path='projects/{project}/global/machineImages', request_field='machineImage', request_type_name='ComputeMachineImagesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of machine images that are contained within the specified project.

      Args:
        request: (ComputeMachineImagesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MachineImageList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.machineImages.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/machineImages', request_field='', request_type_name='ComputeMachineImagesListRequest', response_type_name='MachineImageList', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeMachineImagesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.machineImages.setIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/machineImages/{resource}/setIamPolicy', request_field='globalSetPolicyRequest', request_type_name='ComputeMachineImagesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeMachineImagesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.machineImages.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/machineImages/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeMachineImagesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)