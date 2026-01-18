from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
class ProjectsLocationsMembershipsService(base_api.BaseApiService):
    """Service class for the projects_locations_memberships resource."""
    _NAME = 'projects_locations_memberships'

    def __init__(self, client):
        super(GkehubV1beta.ProjectsLocationsMembershipsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Membership. **This is currently only supported for GKE clusters on Google Cloud**. To register other clusters, follow the instructions at https://cloud.google.com/anthos/multicluster-management/connect/registering-a-cluster.

      Args:
        request: (GkehubProjectsLocationsMembershipsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships', http_method='POST', method_id='gkehub.projects.locations.memberships.create', ordered_params=['parent'], path_params=['parent'], query_params=['membershipId', 'requestId'], relative_path='v1beta/{+parent}/memberships', request_field='membership', request_type_name='GkehubProjectsLocationsMembershipsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Removes a Membership. **This is currently only supported for GKE clusters on Google Cloud**. To unregister other clusters, follow the instructions at https://cloud.google.com/anthos/multicluster-management/connect/unregistering-a-cluster.

      Args:
        request: (GkehubProjectsLocationsMembershipsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}', http_method='DELETE', method_id='gkehub.projects.locations.memberships.delete', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsMembershipsDeleteRequest', response_type_name='Operation', supports_download=False)

    def GenerateConnectManifest(self, request, global_params=None):
        """Generates the manifest for deployment of the GKE connect agent. **This method is used internally by Google-provided libraries.** Most clients should not need to call this method directly.

      Args:
        request: (GkehubProjectsLocationsMembershipsGenerateConnectManifestRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateConnectManifestResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateConnectManifest')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateConnectManifest.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}:generateConnectManifest', http_method='GET', method_id='gkehub.projects.locations.memberships.generateConnectManifest', ordered_params=['name'], path_params=['name'], query_params=['cpuRequest', 'imagePullSecretContent', 'isUpgrade', 'memLimit', 'memRequest', 'namespace', 'proxy', 'registry', 'version'], relative_path='v1beta/{+name}:generateConnectManifest', request_field='', request_type_name='GkehubProjectsLocationsMembershipsGenerateConnectManifestRequest', response_type_name='GenerateConnectManifestResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a Membership.

      Args:
        request: (GkehubProjectsLocationsMembershipsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Membership) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}', http_method='GET', method_id='gkehub.projects.locations.memberships.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsMembershipsGetRequest', response_type_name='Membership', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkehubProjectsLocationsMembershipsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}:getIamPolicy', http_method='GET', method_id='gkehub.projects.locations.memberships.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta/{+resource}:getIamPolicy', request_field='', request_type_name='GkehubProjectsLocationsMembershipsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Memberships in a given project and location.

      Args:
        request: (GkehubProjectsLocationsMembershipsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships', http_method='GET', method_id='gkehub.projects.locations.memberships.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/memberships', request_field='', request_type_name='GkehubProjectsLocationsMembershipsListRequest', response_type_name='ListMembershipsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing Membership.

      Args:
        request: (GkehubProjectsLocationsMembershipsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}', http_method='PATCH', method_id='gkehub.projects.locations.memberships.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1beta/{+name}', request_field='membership', request_type_name='GkehubProjectsLocationsMembershipsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkehubProjectsLocationsMembershipsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}:setIamPolicy', http_method='POST', method_id='gkehub.projects.locations.memberships.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkehubProjectsLocationsMembershipsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkehubProjectsLocationsMembershipsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}:testIamPermissions', http_method='POST', method_id='gkehub.projects.locations.memberships.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkehubProjectsLocationsMembershipsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)