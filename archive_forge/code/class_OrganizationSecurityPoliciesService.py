from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class OrganizationSecurityPoliciesService(base_api.BaseApiService):
    """Service class for the organizationSecurityPolicies resource."""
    _NAME = 'organizationSecurityPolicies'

    def __init__(self, client):
        super(ComputeBeta.OrganizationSecurityPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def AddAssociation(self, request, global_params=None):
        """Inserts an association for the specified security policy.

      Args:
        request: (ComputeOrganizationSecurityPoliciesAddAssociationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddAssociation')
        return self._RunMethod(config, request, global_params=global_params)
    AddAssociation.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.organizationSecurityPolicies.addAssociation', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['replaceExistingAssociation', 'requestId'], relative_path='locations/global/securityPolicies/{securityPolicy}/addAssociation', request_field='securityPolicyAssociation', request_type_name='ComputeOrganizationSecurityPoliciesAddAssociationRequest', response_type_name='Operation', supports_download=False)

    def AddRule(self, request, global_params=None):
        """Inserts a rule into a security policy.

      Args:
        request: (ComputeOrganizationSecurityPoliciesAddRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddRule')
        return self._RunMethod(config, request, global_params=global_params)
    AddRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.organizationSecurityPolicies.addRule', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['requestId'], relative_path='locations/global/securityPolicies/{securityPolicy}/addRule', request_field='securityPolicyRule', request_type_name='ComputeOrganizationSecurityPoliciesAddRuleRequest', response_type_name='Operation', supports_download=False)

    def CopyRules(self, request, global_params=None):
        """Copies rules to the specified security policy.

      Args:
        request: (ComputeOrganizationSecurityPoliciesCopyRulesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CopyRules')
        return self._RunMethod(config, request, global_params=global_params)
    CopyRules.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.organizationSecurityPolicies.copyRules', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['requestId', 'sourceSecurityPolicy'], relative_path='locations/global/securityPolicies/{securityPolicy}/copyRules', request_field='', request_type_name='ComputeOrganizationSecurityPoliciesCopyRulesRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified policy.

      Args:
        request: (ComputeOrganizationSecurityPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.organizationSecurityPolicies.delete', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['requestId'], relative_path='locations/global/securityPolicies/{securityPolicy}', request_field='', request_type_name='ComputeOrganizationSecurityPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """List all of the ordered rules present in a single specified policy.

      Args:
        request: (ComputeOrganizationSecurityPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.organizationSecurityPolicies.get', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=[], relative_path='locations/global/securityPolicies/{securityPolicy}', request_field='', request_type_name='ComputeOrganizationSecurityPoliciesGetRequest', response_type_name='SecurityPolicy', supports_download=False)

    def GetAssociation(self, request, global_params=None):
        """Gets an association with the specified name.

      Args:
        request: (ComputeOrganizationSecurityPoliciesGetAssociationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicyAssociation) The response message.
      """
        config = self.GetMethodConfig('GetAssociation')
        return self._RunMethod(config, request, global_params=global_params)
    GetAssociation.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.organizationSecurityPolicies.getAssociation', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['name'], relative_path='locations/global/securityPolicies/{securityPolicy}/getAssociation', request_field='', request_type_name='ComputeOrganizationSecurityPoliciesGetAssociationRequest', response_type_name='SecurityPolicyAssociation', supports_download=False)

    def GetRule(self, request, global_params=None):
        """Gets a rule at the specified priority.

      Args:
        request: (ComputeOrganizationSecurityPoliciesGetRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicyRule) The response message.
      """
        config = self.GetMethodConfig('GetRule')
        return self._RunMethod(config, request, global_params=global_params)
    GetRule.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.organizationSecurityPolicies.getRule', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['priority'], relative_path='locations/global/securityPolicies/{securityPolicy}/getRule', request_field='', request_type_name='ComputeOrganizationSecurityPoliciesGetRuleRequest', response_type_name='SecurityPolicyRule', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new policy in the specified project using the data included in the request.

      Args:
        request: (ComputeOrganizationSecurityPoliciesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.organizationSecurityPolicies.insert', ordered_params=[], path_params=[], query_params=['parentId', 'requestId'], relative_path='locations/global/securityPolicies', request_field='securityPolicy', request_type_name='ComputeOrganizationSecurityPoliciesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """List all the policies that have been configured for the specified project.

      Args:
        request: (ComputeOrganizationSecurityPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.organizationSecurityPolicies.list', ordered_params=[], path_params=[], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'parentId', 'returnPartialSuccess'], relative_path='locations/global/securityPolicies', request_field='', request_type_name='ComputeOrganizationSecurityPoliciesListRequest', response_type_name='SecurityPolicyList', supports_download=False)

    def ListAssociations(self, request, global_params=None):
        """Lists associations of a specified target, i.e., organization or folder.

      Args:
        request: (ComputeOrganizationSecurityPoliciesListAssociationsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OrganizationSecurityPoliciesListAssociationsResponse) The response message.
      """
        config = self.GetMethodConfig('ListAssociations')
        return self._RunMethod(config, request, global_params=global_params)
    ListAssociations.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.organizationSecurityPolicies.listAssociations', ordered_params=[], path_params=[], query_params=['targetResource'], relative_path='locations/global/securityPolicies/listAssociations', request_field='', request_type_name='ComputeOrganizationSecurityPoliciesListAssociationsRequest', response_type_name='OrganizationSecurityPoliciesListAssociationsResponse', supports_download=False)

    def Move(self, request, global_params=None):
        """Moves the specified security policy.

      Args:
        request: (ComputeOrganizationSecurityPoliciesMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Move')
        return self._RunMethod(config, request, global_params=global_params)
    Move.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.organizationSecurityPolicies.move', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['parentId', 'requestId'], relative_path='locations/global/securityPolicies/{securityPolicy}/move', request_field='', request_type_name='ComputeOrganizationSecurityPoliciesMoveRequest', response_type_name='Operation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified policy with the data included in the request.

      Args:
        request: (ComputeOrganizationSecurityPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.organizationSecurityPolicies.patch', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['requestId'], relative_path='locations/global/securityPolicies/{securityPolicy}', request_field='securityPolicyResource', request_type_name='ComputeOrganizationSecurityPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def PatchRule(self, request, global_params=None):
        """Patches a rule at the specified priority.

      Args:
        request: (ComputeOrganizationSecurityPoliciesPatchRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PatchRule')
        return self._RunMethod(config, request, global_params=global_params)
    PatchRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.organizationSecurityPolicies.patchRule', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['priority', 'requestId'], relative_path='locations/global/securityPolicies/{securityPolicy}/patchRule', request_field='securityPolicyRule', request_type_name='ComputeOrganizationSecurityPoliciesPatchRuleRequest', response_type_name='Operation', supports_download=False)

    def RemoveAssociation(self, request, global_params=None):
        """Removes an association for the specified security policy.

      Args:
        request: (ComputeOrganizationSecurityPoliciesRemoveAssociationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveAssociation')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveAssociation.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.organizationSecurityPolicies.removeAssociation', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['name', 'requestId'], relative_path='locations/global/securityPolicies/{securityPolicy}/removeAssociation', request_field='', request_type_name='ComputeOrganizationSecurityPoliciesRemoveAssociationRequest', response_type_name='Operation', supports_download=False)

    def RemoveRule(self, request, global_params=None):
        """Deletes a rule at the specified priority.

      Args:
        request: (ComputeOrganizationSecurityPoliciesRemoveRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveRule')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.organizationSecurityPolicies.removeRule', ordered_params=['securityPolicy'], path_params=['securityPolicy'], query_params=['priority', 'requestId'], relative_path='locations/global/securityPolicies/{securityPolicy}/removeRule', request_field='', request_type_name='ComputeOrganizationSecurityPoliciesRemoveRuleRequest', response_type_name='Operation', supports_download=False)