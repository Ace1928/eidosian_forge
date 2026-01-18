from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securityposture.v1alpha import securityposture_v1alpha_messages as messages
class OrganizationsLocationsPosturesService(base_api.BaseApiService):
    """Service class for the organizations_locations_postures resource."""
    _NAME = 'organizations_locations_postures'

    def __init__(self, client):
        super(SecuritypostureV1alpha.OrganizationsLocationsPosturesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Posture resource. If a Posture with the specified name already exists in the specified organization and location, the long running operation returns a ALREADY_EXISTS error.

      Args:
        request: (SecuritypostureOrganizationsLocationsPosturesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/postures', http_method='POST', method_id='securityposture.organizations.locations.postures.create', ordered_params=['parent'], path_params=['parent'], query_params=['postureId'], relative_path='v1alpha/{+parent}/postures', request_field='posture', request_type_name='SecuritypostureOrganizationsLocationsPosturesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes all the revisions of a resource. A posture can only be deleted when none of the revisions are deployed to any workload.

      Args:
        request: (SecuritypostureOrganizationsLocationsPosturesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/postures/{posturesId}', http_method='DELETE', method_id='securityposture.organizations.locations.postures.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1alpha/{+name}', request_field='', request_type_name='SecuritypostureOrganizationsLocationsPosturesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Extract(self, request, global_params=None):
        """Extracts existing policies on a workload as a posture. If a Posture on the given workload already exists, the long running operation returns a ALREADY_EXISTS error.

      Args:
        request: (SecuritypostureOrganizationsLocationsPosturesExtractRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Extract')
        return self._RunMethod(config, request, global_params=global_params)
    Extract.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/postures:extract', http_method='POST', method_id='securityposture.organizations.locations.postures.extract', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/postures:extract', request_field='extractPostureRequest', request_type_name='SecuritypostureOrganizationsLocationsPosturesExtractRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a posture in a given organization and location. User must provide revision_id to retrieve a specific revision of the resource. NOT_FOUND error is returned if the revision_id or the Posture name does not exist. In case revision_id is not provided then the latest Posture revision by UpdateTime is returned.

      Args:
        request: (SecuritypostureOrganizationsLocationsPosturesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Posture) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/postures/{posturesId}', http_method='GET', method_id='securityposture.organizations.locations.postures.get', ordered_params=['name'], path_params=['name'], query_params=['revisionId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='SecuritypostureOrganizationsLocationsPosturesGetRequest', response_type_name='Posture', supports_download=False)

    def List(self, request, global_params=None):
        """========================== Postures ========================== Lists Postures in a given organization and location. In case a posture has multiple revisions, the latest revision as per UpdateTime will be returned.

      Args:
        request: (SecuritypostureOrganizationsLocationsPosturesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPosturesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/postures', http_method='GET', method_id='securityposture.organizations.locations.postures.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/postures', request_field='', request_type_name='SecuritypostureOrganizationsLocationsPosturesListRequest', response_type_name='ListPosturesResponse', supports_download=False)

    def ListRevisions(self, request, global_params=None):
        """Lists revisions of a Posture in a given organization and location.

      Args:
        request: (SecuritypostureOrganizationsLocationsPosturesListRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPostureRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('ListRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    ListRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/postures/{posturesId}:listRevisions', http_method='GET', method_id='securityposture.organizations.locations.postures.listRevisions', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+name}:listRevisions', request_field='', request_type_name='SecuritypostureOrganizationsLocationsPosturesListRevisionsRequest', response_type_name='ListPostureRevisionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing Posture. A new revision of the posture will be created if the revision to be updated is currently deployed on a workload. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the Posture does not exist. Returns a `google.rpc.Status` with `google.rpc.Code.ABORTED` if the etag supplied in the request does not match the persisted etag of the Posture. Updatable fields are state, description and policy_sets. State update operation cannot be clubbed with update of description and policy_sets. An ACTIVE posture can be updated to both DRAFT or DEPRECATED states. Postures in DRAFT or DEPRECATED states can only be updated to ACTIVE state.

      Args:
        request: (SecuritypostureOrganizationsLocationsPosturesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/postures/{posturesId}', http_method='PATCH', method_id='securityposture.organizations.locations.postures.patch', ordered_params=['name'], path_params=['name'], query_params=['revisionId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='posture', request_type_name='SecuritypostureOrganizationsLocationsPosturesPatchRequest', response_type_name='Operation', supports_download=False)