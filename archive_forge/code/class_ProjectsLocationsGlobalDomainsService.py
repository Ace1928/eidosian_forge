from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
class ProjectsLocationsGlobalDomainsService(base_api.BaseApiService):
    """Service class for the projects_locations_global_domains resource."""
    _NAME = 'projects_locations_global_domains'

    def __init__(self, client):
        super(ManagedidentitiesV1.ProjectsLocationsGlobalDomainsService, self).__init__(client)
        self._upload_configs = {}

    def AttachTrust(self, request, global_params=None):
        """Adds an AD trust to a domain.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsAttachTrustRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AttachTrust')
        return self._RunMethod(config, request, global_params=global_params)
    AttachTrust.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:attachTrust', http_method='POST', method_id='managedidentities.projects.locations.global.domains.attachTrust', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:attachTrust', request_field='attachTrustRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsAttachTrustRequest', response_type_name='Operation', supports_download=False)

    def CheckMigrationPermission(self, request, global_params=None):
        """CheckMigrationPermission API gets the current state of DomainMigration.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsCheckMigrationPermissionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckMigrationPermissionResponse) The response message.
      """
        config = self.GetMethodConfig('CheckMigrationPermission')
        return self._RunMethod(config, request, global_params=global_params)
    CheckMigrationPermission.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:checkMigrationPermission', http_method='POST', method_id='managedidentities.projects.locations.global.domains.checkMigrationPermission', ordered_params=['domain'], path_params=['domain'], query_params=[], relative_path='v1/{+domain}:checkMigrationPermission', request_field='checkMigrationPermissionRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsCheckMigrationPermissionRequest', response_type_name='CheckMigrationPermissionResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a Microsoft AD domain.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains', http_method='POST', method_id='managedidentities.projects.locations.global.domains.create', ordered_params=['parent'], path_params=['parent'], query_params=['domainName'], relative_path='v1/{+parent}/domains', request_field='domain', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a domain.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}', http_method='DELETE', method_id='managedidentities.projects.locations.global.domains.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsDeleteRequest', response_type_name='Operation', supports_download=False)

    def DetachTrust(self, request, global_params=None):
        """Removes an AD trust.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsDetachTrustRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DetachTrust')
        return self._RunMethod(config, request, global_params=global_params)
    DetachTrust.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:detachTrust', http_method='POST', method_id='managedidentities.projects.locations.global.domains.detachTrust', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:detachTrust', request_field='detachTrustRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsDetachTrustRequest', response_type_name='Operation', supports_download=False)

    def DisableMigration(self, request, global_params=None):
        """Disable Domain Migration.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsDisableMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DisableMigration')
        return self._RunMethod(config, request, global_params=global_params)
    DisableMigration.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:disableMigration', http_method='POST', method_id='managedidentities.projects.locations.global.domains.disableMigration', ordered_params=['domain'], path_params=['domain'], query_params=[], relative_path='v1/{+domain}:disableMigration', request_field='disableMigrationRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsDisableMigrationRequest', response_type_name='Operation', supports_download=False)

    def DomainJoinMachine(self, request, global_params=None):
        """DomainJoinMachine API joins a Compute Engine VM to the domain.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsDomainJoinMachineRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DomainJoinMachineResponse) The response message.
      """
        config = self.GetMethodConfig('DomainJoinMachine')
        return self._RunMethod(config, request, global_params=global_params)
    DomainJoinMachine.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:domainJoinMachine', http_method='POST', method_id='managedidentities.projects.locations.global.domains.domainJoinMachine', ordered_params=['domain'], path_params=['domain'], query_params=[], relative_path='v1/{+domain}:domainJoinMachine', request_field='domainJoinMachineRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsDomainJoinMachineRequest', response_type_name='DomainJoinMachineResponse', supports_download=False)

    def EnableMigration(self, request, global_params=None):
        """Enable Domain Migration.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsEnableMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('EnableMigration')
        return self._RunMethod(config, request, global_params=global_params)
    EnableMigration.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:enableMigration', http_method='POST', method_id='managedidentities.projects.locations.global.domains.enableMigration', ordered_params=['domain'], path_params=['domain'], query_params=[], relative_path='v1/{+domain}:enableMigration', request_field='enableMigrationRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsEnableMigrationRequest', response_type_name='Operation', supports_download=False)

    def ExtendSchema(self, request, global_params=None):
        """Extend Schema for Domain.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsExtendSchemaRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ExtendSchema')
        return self._RunMethod(config, request, global_params=global_params)
    ExtendSchema.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:extendSchema', http_method='POST', method_id='managedidentities.projects.locations.global.domains.extendSchema', ordered_params=['domain'], path_params=['domain'], query_params=[], relative_path='v1/{+domain}:extendSchema', request_field='extendSchemaRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsExtendSchemaRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a domain.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Domain) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}', http_method='GET', method_id='managedidentities.projects.locations.global.domains.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsGetRequest', response_type_name='Domain', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:getIamPolicy', http_method='GET', method_id='managedidentities.projects.locations.global.domains.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def GetLdapssettings(self, request, global_params=None):
        """Gets the domain ldaps settings.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsGetLdapssettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LDAPSSettings) The response message.
      """
        config = self.GetMethodConfig('GetLdapssettings')
        return self._RunMethod(config, request, global_params=global_params)
    GetLdapssettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}/ldapssettings', http_method='GET', method_id='managedidentities.projects.locations.global.domains.getLdapssettings', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/ldapssettings', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsGetLdapssettingsRequest', response_type_name='LDAPSSettings', supports_download=False)

    def List(self, request, global_params=None):
        """Lists domains in a project.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDomainsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains', http_method='GET', method_id='managedidentities.projects.locations.global.domains.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/domains', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsListRequest', response_type_name='ListDomainsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the metadata and configuration of a domain.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}', http_method='PATCH', method_id='managedidentities.projects.locations.global.domains.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='domain', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsPatchRequest', response_type_name='Operation', supports_download=False)

    def ReconfigureTrust(self, request, global_params=None):
        """Updates the DNS conditional forwarder.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsReconfigureTrustRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ReconfigureTrust')
        return self._RunMethod(config, request, global_params=global_params)
    ReconfigureTrust.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:reconfigureTrust', http_method='POST', method_id='managedidentities.projects.locations.global.domains.reconfigureTrust', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:reconfigureTrust', request_field='reconfigureTrustRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsReconfigureTrustRequest', response_type_name='Operation', supports_download=False)

    def ResetAdminPassword(self, request, global_params=None):
        """Resets a domain's administrator password.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsResetAdminPasswordRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResetAdminPasswordResponse) The response message.
      """
        config = self.GetMethodConfig('ResetAdminPassword')
        return self._RunMethod(config, request, global_params=global_params)
    ResetAdminPassword.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:resetAdminPassword', http_method='POST', method_id='managedidentities.projects.locations.global.domains.resetAdminPassword', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:resetAdminPassword', request_field='resetAdminPasswordRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsResetAdminPasswordRequest', response_type_name='ResetAdminPasswordResponse', supports_download=False)

    def Restore(self, request, global_params=None):
        """RestoreDomain restores domain backup mentioned in the RestoreDomainRequest.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Restore')
        return self._RunMethod(config, request, global_params=global_params)
    Restore.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:restore', http_method='POST', method_id='managedidentities.projects.locations.global.domains.restore', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:restore', request_field='restoreDomainRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsRestoreRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:setIamPolicy', http_method='POST', method_id='managedidentities.projects.locations.global.domains.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:testIamPermissions', http_method='POST', method_id='managedidentities.projects.locations.global.domains.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def UpdateLdapssettings(self, request, global_params=None):
        """Patches a single ldaps settings.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsUpdateLdapssettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateLdapssettings')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateLdapssettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}/ldapssettings', http_method='PATCH', method_id='managedidentities.projects.locations.global.domains.updateLdapssettings', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}/ldapssettings', request_field='lDAPSSettings', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsUpdateLdapssettingsRequest', response_type_name='Operation', supports_download=False)

    def ValidateTrust(self, request, global_params=None):
        """Validates a trust state, that the target domain is reachable, and that the target domain is able to accept incoming trust requests.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsValidateTrustRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ValidateTrust')
        return self._RunMethod(config, request, global_params=global_params)
    ValidateTrust.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}:validateTrust', http_method='POST', method_id='managedidentities.projects.locations.global.domains.validateTrust', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:validateTrust', request_field='validateTrustRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsValidateTrustRequest', response_type_name='Operation', supports_download=False)