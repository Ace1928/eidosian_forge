from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class TargetInstancesService(base_api.BaseApiService):
    """Service class for the targetInstances resource."""
    _NAME = 'targetInstances'

    def __init__(self, client):
        super(ComputeBeta.TargetInstancesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of target instances. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeTargetInstancesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetInstanceAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetInstances.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/targetInstances', request_field='', request_type_name='ComputeTargetInstancesAggregatedListRequest', response_type_name='TargetInstanceAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified TargetInstance resource.

      Args:
        request: (ComputeTargetInstancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.targetInstances.delete', ordered_params=['project', 'zone', 'targetInstance'], path_params=['project', 'targetInstance', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/targetInstances/{targetInstance}', request_field='', request_type_name='ComputeTargetInstancesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified TargetInstance resource.

      Args:
        request: (ComputeTargetInstancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetInstance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetInstances.get', ordered_params=['project', 'zone', 'targetInstance'], path_params=['project', 'targetInstance', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/targetInstances/{targetInstance}', request_field='', request_type_name='ComputeTargetInstancesGetRequest', response_type_name='TargetInstance', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a TargetInstance resource in the specified project and zone using the data included in the request.

      Args:
        request: (ComputeTargetInstancesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetInstances.insert', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/targetInstances', request_field='targetInstance', request_type_name='ComputeTargetInstancesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of TargetInstance resources available to the specified project and zone.

      Args:
        request: (ComputeTargetInstancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetInstanceList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetInstances.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/targetInstances', request_field='', request_type_name='ComputeTargetInstancesListRequest', response_type_name='TargetInstanceList', supports_download=False)

    def SetSecurityPolicy(self, request, global_params=None):
        """Sets the Google Cloud Armor security policy for the specified target instance. For more information, see Google Cloud Armor Overview.

      Args:
        request: (ComputeTargetInstancesSetSecurityPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetSecurityPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetSecurityPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetInstances.setSecurityPolicy', ordered_params=['project', 'zone', 'targetInstance'], path_params=['project', 'targetInstance', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/targetInstances/{targetInstance}/setSecurityPolicy', request_field='securityPolicyReference', request_type_name='ComputeTargetInstancesSetSecurityPolicyRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeTargetInstancesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetInstances.testIamPermissions', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/targetInstances/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeTargetInstancesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)