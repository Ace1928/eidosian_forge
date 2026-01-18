from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class BackendServicesService(base_api.BaseApiService):
    """Service class for the backendServices resource."""
    _NAME = 'backendServices'

    def __init__(self, client):
        super(ComputeBeta.BackendServicesService, self).__init__(client)
        self._upload_configs = {}

    def AddSignedUrlKey(self, request, global_params=None):
        """Adds a key for validating requests with signed URLs for this backend service.

      Args:
        request: (ComputeBackendServicesAddSignedUrlKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddSignedUrlKey')
        return self._RunMethod(config, request, global_params=global_params)
    AddSignedUrlKey.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendServices.addSignedUrlKey', ordered_params=['project', 'backendService'], path_params=['backendService', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendServices/{backendService}/addSignedUrlKey', request_field='signedUrlKey', request_type_name='ComputeBackendServicesAddSignedUrlKeyRequest', response_type_name='Operation', supports_download=False)

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all BackendService resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeBackendServicesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendServiceAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.backendServices.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/backendServices', request_field='', request_type_name='ComputeBackendServicesAggregatedListRequest', response_type_name='BackendServiceAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified BackendService resource.

      Args:
        request: (ComputeBackendServicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.backendServices.delete', ordered_params=['project', 'backendService'], path_params=['backendService', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendServices/{backendService}', request_field='', request_type_name='ComputeBackendServicesDeleteRequest', response_type_name='Operation', supports_download=False)

    def DeleteSignedUrlKey(self, request, global_params=None):
        """Deletes a key for validating requests with signed URLs for this backend service.

      Args:
        request: (ComputeBackendServicesDeleteSignedUrlKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DeleteSignedUrlKey')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteSignedUrlKey.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendServices.deleteSignedUrlKey', ordered_params=['project', 'backendService', 'keyName'], path_params=['backendService', 'project'], query_params=['keyName', 'requestId'], relative_path='projects/{project}/global/backendServices/{backendService}/deleteSignedUrlKey', request_field='', request_type_name='ComputeBackendServicesDeleteSignedUrlKeyRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified BackendService resource.

      Args:
        request: (ComputeBackendServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendService) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.backendServices.get', ordered_params=['project', 'backendService'], path_params=['backendService', 'project'], query_params=[], relative_path='projects/{project}/global/backendServices/{backendService}', request_field='', request_type_name='ComputeBackendServicesGetRequest', response_type_name='BackendService', supports_download=False)

    def GetHealth(self, request, global_params=None):
        """Gets the most recent health check results for this BackendService. Example request body: { "group": "/zones/us-east1-b/instanceGroups/lb-backend-example" }.

      Args:
        request: (ComputeBackendServicesGetHealthRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendServiceGroupHealth) The response message.
      """
        config = self.GetMethodConfig('GetHealth')
        return self._RunMethod(config, request, global_params=global_params)
    GetHealth.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendServices.getHealth', ordered_params=['project', 'backendService'], path_params=['backendService', 'project'], query_params=[], relative_path='projects/{project}/global/backendServices/{backendService}/getHealth', request_field='resourceGroupReference', request_type_name='ComputeBackendServicesGetHealthRequest', response_type_name='BackendServiceGroupHealth', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeBackendServicesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.backendServices.getIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/global/backendServices/{resource}/getIamPolicy', request_field='', request_type_name='ComputeBackendServicesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a BackendService resource in the specified project using the data included in the request. For more information, see Backend services overview .

      Args:
        request: (ComputeBackendServicesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendServices.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/backendServices', request_field='backendService', request_type_name='ComputeBackendServicesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of BackendService resources available to the specified project.

      Args:
        request: (ComputeBackendServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendServiceList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.backendServices.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/backendServices', request_field='', request_type_name='ComputeBackendServicesListRequest', response_type_name='BackendServiceList', supports_download=False)

    def ListUsable(self, request, global_params=None):
        """Retrieves an aggregated list of all usable backend services in the specified project.

      Args:
        request: (ComputeBackendServicesListUsableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendServiceListUsable) The response message.
      """
        config = self.GetMethodConfig('ListUsable')
        return self._RunMethod(config, request, global_params=global_params)
    ListUsable.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.backendServices.listUsable', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/backendServices/listUsable', request_field='', request_type_name='ComputeBackendServicesListUsableRequest', response_type_name='BackendServiceListUsable', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified BackendService resource with the data included in the request. For more information, see Backend services overview. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeBackendServicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.backendServices.patch', ordered_params=['project', 'backendService'], path_params=['backendService', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendServices/{backendService}', request_field='backendServiceResource', request_type_name='ComputeBackendServicesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetEdgeSecurityPolicy(self, request, global_params=None):
        """Sets the edge security policy for the specified backend service.

      Args:
        request: (ComputeBackendServicesSetEdgeSecurityPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetEdgeSecurityPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetEdgeSecurityPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendServices.setEdgeSecurityPolicy', ordered_params=['project', 'backendService'], path_params=['backendService', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendServices/{backendService}/setEdgeSecurityPolicy', request_field='securityPolicyReference', request_type_name='ComputeBackendServicesSetEdgeSecurityPolicyRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeBackendServicesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendServices.setIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/backendServices/{resource}/setIamPolicy', request_field='globalSetPolicyRequest', request_type_name='ComputeBackendServicesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetSecurityPolicy(self, request, global_params=None):
        """Sets the Google Cloud Armor security policy for the specified backend service. For more information, see Google Cloud Armor Overview.

      Args:
        request: (ComputeBackendServicesSetSecurityPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetSecurityPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetSecurityPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendServices.setSecurityPolicy', ordered_params=['project', 'backendService'], path_params=['backendService', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendServices/{backendService}/setSecurityPolicy', request_field='securityPolicyReference', request_type_name='ComputeBackendServicesSetSecurityPolicyRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeBackendServicesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendServices.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/backendServices/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeBackendServicesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified BackendService resource with the data included in the request. For more information, see Backend services overview.

      Args:
        request: (ComputeBackendServicesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.backendServices.update', ordered_params=['project', 'backendService'], path_params=['backendService', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendServices/{backendService}', request_field='backendServiceResource', request_type_name='ComputeBackendServicesUpdateRequest', response_type_name='Operation', supports_download=False)