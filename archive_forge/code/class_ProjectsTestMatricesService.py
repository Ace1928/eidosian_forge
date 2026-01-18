from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.testing.v1 import testing_v1_messages as messages
class ProjectsTestMatricesService(base_api.BaseApiService):
    """Service class for the projects_testMatrices resource."""
    _NAME = 'projects_testMatrices'

    def __init__(self, client):
        super(TestingV1.ProjectsTestMatricesService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels unfinished test executions in a test matrix. This call returns immediately and cancellation proceeds asynchronously. If the matrix is already final, this operation will have no effect. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the Test Matrix does not exist.

      Args:
        request: (TestingProjectsTestMatricesCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CancelTestMatrixResponse) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='testing.projects.testMatrices.cancel', ordered_params=['projectId', 'testMatrixId'], path_params=['projectId', 'testMatrixId'], query_params=[], relative_path='v1/projects/{projectId}/testMatrices/{testMatrixId}:cancel', request_field='', request_type_name='TestingProjectsTestMatricesCancelRequest', response_type_name='CancelTestMatrixResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates and runs a matrix of tests according to the given specifications. Unsupported environments will be returned in the state UNSUPPORTED. A test matrix is limited to use at most 2000 devices in parallel. The returned matrix will not yet contain the executions that will be created for this matrix. Execution creation happens later on and will require a call to GetTestMatrix. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write to project - INVALID_ARGUMENT - if the request is malformed or if the matrix tries to use too many simultaneous devices.

      Args:
        request: (TestingProjectsTestMatricesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestMatrix) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='testing.projects.testMatrices.create', ordered_params=['projectId'], path_params=['projectId'], query_params=['requestId'], relative_path='v1/projects/{projectId}/testMatrices', request_field='testMatrix', request_type_name='TestingProjectsTestMatricesCreateRequest', response_type_name='TestMatrix', supports_download=False)

    def Get(self, request, global_params=None):
        """Checks the status of a test matrix and the executions once they are created. The test matrix will contain the list of test executions to run if and only if the resultStorage.toolResultsExecution fields have been populated. Note: Flaky test executions may be added to the matrix at a later stage. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the Test Matrix does not exist.

      Args:
        request: (TestingProjectsTestMatricesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestMatrix) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='testing.projects.testMatrices.get', ordered_params=['projectId', 'testMatrixId'], path_params=['projectId', 'testMatrixId'], query_params=[], relative_path='v1/projects/{projectId}/testMatrices/{testMatrixId}', request_field='', request_type_name='TestingProjectsTestMatricesGetRequest', response_type_name='TestMatrix', supports_download=False)