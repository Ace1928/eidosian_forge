from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
class OrgSecurityPolicyRule(OrgSecurityPolicy):
    """Abstracts Organization SecurityPolicy Rule."""

    def __init__(self, ref=None, compute_client=None, resources=None, version='beta'):
        super(OrgSecurityPolicyRule, self).__init__(ref=ref, compute_client=compute_client, resources=resources, version=version)

    def _MakeCreateRuleRequestTuple(self, security_policy=None, security_policy_rule=None):
        return (self._client.organizationSecurityPolicies, 'AddRule', self._messages.ComputeOrganizationSecurityPoliciesAddRuleRequest(securityPolicy=security_policy, securityPolicyRule=security_policy_rule))

    def _MakeDeleteRuleRequestTuple(self, priority=None, security_policy=None):
        return (self._client.organizationSecurityPolicies, 'RemoveRule', self._messages.ComputeOrganizationSecurityPoliciesRemoveRuleRequest(securityPolicy=security_policy, priority=priority))

    def _MakeDescribeRuleRequestTuple(self, priority=None, security_policy=None):
        return (self._client.organizationSecurityPolicies, 'GetRule', self._messages.ComputeOrganizationSecurityPoliciesGetRuleRequest(securityPolicy=security_policy, priority=priority))

    def _MakeUpdateRuleRequestTuple(self, priority=None, security_policy=None, security_policy_rule=None):
        return (self._client.organizationSecurityPolicies, 'PatchRule', self._messages.ComputeOrganizationSecurityPoliciesPatchRuleRequest(priority=priority, securityPolicy=security_policy, securityPolicyRule=security_policy_rule))

    def Create(self, security_policy=None, security_policy_rule=None, batch_mode=False, only_generate_request=False):
        """Sends request to create a security policy rule."""
        if batch_mode:
            requests = [self._MakeCreateRuleRequestTuple(security_policy=security_policy, security_policy_rule=security_policy_rule)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.AddRule(self._MakeCreateRuleRequestTuple(security_policy=security_policy, security_policy_rule=security_policy_rule)[2])
        return self.WaitOperation(op_res, message='Add a rule of the organization Security Policy.')

    def Delete(self, priority=None, security_policy_id=None, batch_mode=False, only_generate_request=False):
        """Sends request to delete a security policy rule."""
        if batch_mode:
            requests = [self._MakeDeleteRuleRequestTuple(priority=priority, security_policy=security_policy_id)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.RemoveRule(self._MakeDeleteRuleRequestTuple(priority=priority, security_policy=security_policy_id)[2])
        return self.WaitOperation(op_res, message='Delete a rule of the organization Security Policy.')

    def Describe(self, priority=None, security_policy_id=None, batch_mode=False, only_generate_request=False):
        """Sends request to describe a security policy rule."""
        if batch_mode:
            requests = [self._MakeDescribeRuleRequestTuple(priority=priority, security_policy=security_policy_id)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        return self._service.GetRule(self._MakeDescribeRuleRequestTuple(priority=priority, security_policy=security_policy_id)[2])

    def Update(self, priority=None, security_policy=None, security_policy_rule=None, batch_mode=False, only_generate_request=False):
        """Sends request to update a security policy rule."""
        if batch_mode:
            requests = [self._MakeUpdateRuleRequestTuple(priority=priority, security_policy=security_policy, security_policy_rule=security_policy_rule)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.PatchRule(self._MakeUpdateRuleRequestTuple(priority=priority, security_policy=security_policy, security_policy_rule=security_policy_rule)[2])
        return self.WaitOperation(op_res, message='Update a rule of the organization Security Policy.')