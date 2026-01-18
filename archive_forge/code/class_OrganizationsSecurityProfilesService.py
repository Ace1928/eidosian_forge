from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsSecurityProfilesService(base_api.BaseApiService):
    """Service class for the organizations_securityProfiles resource."""
    _NAME = 'organizations_securityProfiles'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsSecurityProfilesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """CreateSecurityProfile create a new custom security profile.

      Args:
        request: (ApigeeOrganizationsSecurityProfilesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityProfile) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityProfiles', http_method='POST', method_id='apigee.organizations.securityProfiles.create', ordered_params=['parent'], path_params=['parent'], query_params=['securityProfileId'], relative_path='v1/{+parent}/securityProfiles', request_field='googleCloudApigeeV1SecurityProfile', request_type_name='ApigeeOrganizationsSecurityProfilesCreateRequest', response_type_name='GoogleCloudApigeeV1SecurityProfile', supports_download=False)

    def Delete(self, request, global_params=None):
        """DeleteSecurityProfile delete a profile with all its revisions.

      Args:
        request: (ApigeeOrganizationsSecurityProfilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityProfiles/{securityProfilesId}', http_method='DELETE', method_id='apigee.organizations.securityProfiles.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSecurityProfilesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """GetSecurityProfile gets the specified security profile. Returns NOT_FOUND if security profile is not present for the specified organization.

      Args:
        request: (ApigeeOrganizationsSecurityProfilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityProfile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityProfiles/{securityProfilesId}', http_method='GET', method_id='apigee.organizations.securityProfiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSecurityProfilesGetRequest', response_type_name='GoogleCloudApigeeV1SecurityProfile', supports_download=False)

    def List(self, request, global_params=None):
        """ListSecurityProfiles lists all the security profiles associated with the org including attached and unattached profiles.

      Args:
        request: (ApigeeOrganizationsSecurityProfilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListSecurityProfilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityProfiles', http_method='GET', method_id='apigee.organizations.securityProfiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/securityProfiles', request_field='', request_type_name='ApigeeOrganizationsSecurityProfilesListRequest', response_type_name='GoogleCloudApigeeV1ListSecurityProfilesResponse', supports_download=False)

    def ListRevisions(self, request, global_params=None):
        """ListSecurityProfileRevisions lists all the revisions of the security profile.

      Args:
        request: (ApigeeOrganizationsSecurityProfilesListRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListSecurityProfileRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('ListRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    ListRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityProfiles/{securityProfilesId}:listRevisions', http_method='GET', method_id='apigee.organizations.securityProfiles.listRevisions', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+name}:listRevisions', request_field='', request_type_name='ApigeeOrganizationsSecurityProfilesListRevisionsRequest', response_type_name='GoogleCloudApigeeV1ListSecurityProfileRevisionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """UpdateSecurityProfile update the metadata of security profile.

      Args:
        request: (ApigeeOrganizationsSecurityProfilesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityProfile) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityProfiles/{securityProfilesId}', http_method='PATCH', method_id='apigee.organizations.securityProfiles.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1SecurityProfile', request_type_name='ApigeeOrganizationsSecurityProfilesPatchRequest', response_type_name='GoogleCloudApigeeV1SecurityProfile', supports_download=False)