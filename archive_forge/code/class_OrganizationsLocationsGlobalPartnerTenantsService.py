from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
class OrganizationsLocationsGlobalPartnerTenantsService(base_api.BaseApiService):
    """Service class for the organizations_locations_global_partnerTenants resource."""
    _NAME = 'organizations_locations_global_partnerTenants'

    def __init__(self, client):
        super(BeyondcorpV1alpha.OrganizationsLocationsGlobalPartnerTenantsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new BeyondCorp Enterprise partnerTenant in a given organization and can only be called by onboarded BeyondCorp Enterprise partner.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants', http_method='POST', method_id='beyondcorp.organizations.locations.global.partnerTenants.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId'], relative_path='v1alpha/{+parent}/partnerTenants', request_field='googleCloudBeyondcorpPartnerservicesV1alphaPartnerTenant', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single PartnerTenant.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}', http_method='DELETE', method_id='beyondcorp.organizations.locations.global.partnerTenants.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single PartnerTenant.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpPartnerservicesV1alphaPartnerTenant) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}', http_method='GET', method_id='beyondcorp.organizations.locations.global.partnerTenants.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsGetRequest', response_type_name='GoogleCloudBeyondcorpPartnerservicesV1alphaPartnerTenant', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}:getIamPolicy', http_method='GET', method_id='beyondcorp.organizations.locations.global.partnerTenants.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists PartnerTenants in a given organization.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpPartnerservicesV1alphaListPartnerTenantsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants', http_method='GET', method_id='beyondcorp.organizations.locations.global.partnerTenants.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/partnerTenants', request_field='', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsListRequest', response_type_name='GoogleCloudBeyondcorpPartnerservicesV1alphaListPartnerTenantsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a single PartnerTenant.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}', http_method='PATCH', method_id='beyondcorp.organizations.locations.global.partnerTenants.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='googleCloudBeyondcorpPartnerservicesV1alphaPartnerTenant', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}:setIamPolicy', http_method='POST', method_id='beyondcorp.organizations.locations.global.partnerTenants.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}:testIamPermissions', http_method='POST', method_id='beyondcorp.organizations.locations.global.partnerTenants.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)