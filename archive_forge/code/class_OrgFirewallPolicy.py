from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
class OrgFirewallPolicy(object):
    """Abstracts an organization firewall policy resource."""

    def __init__(self, ref=None, compute_client=None, resources=None, version='v1'):
        self.ref = ref
        self._compute_client = compute_client
        self._resources = resources
        self._version = 'v1' if version == 'ga' else version
        self._op_has_project = self._HasProject(OP_COLLECTION_NAME)
        self._api_has_project = self._HasProject(API_COLLECTION_NAME)

    def _HasProject(self, collection):
        collection_info = self._resources.GetCollectionInfo(collection, self._version)
        return 'projects' in collection_info.path or 'projects' in collection_info.base_url

    @property
    def _client(self):
        return self._compute_client.apitools_client

    @property
    def _messages(self):
        return self._compute_client.messages

    @property
    def _service(self):
        return self._client.firewallPolicies

    def _MakeAddAssociationRequestTuple(self, association, firewall_policy_id, replace_existing_association):
        return (self._client.firewallPolicies, 'AddAssociation', self._messages.ComputeFirewallPoliciesAddAssociationRequest(firewallPolicyAssociation=association, firewallPolicy=firewall_policy_id, replaceExistingAssociation=replace_existing_association))

    def _MakeDeleteAssociationRequestTuple(self, firewall_policy_id):
        return (self._client.firewallPolicies, 'RemoveAssociation', self._messages.ComputeFirewallPoliciesRemoveAssociationRequest(name=self.ref.Name(), firewallPolicy=firewall_policy_id))

    def _MakeListAssociationsRequestTuple(self, target_resource):
        return (self._client.firewallPolicies, 'ListAssociations', self._messages.ComputeFirewallPoliciesListAssociationsRequest(targetResource=target_resource))

    def _MakeDeleteRequestTuple(self, fp_id=None):
        return (self._client.firewallPolicies, 'Delete', self._messages.ComputeFirewallPoliciesDeleteRequest(firewallPolicy=fp_id))

    def _MakeUpdateRequestTuple(self, fp_id=None, firewall_policy=None):
        if fp_id:
            return (self._client.firewallPolicies, 'Patch', self._messages.ComputeFirewallPoliciesPatchRequest(firewallPolicy=fp_id, firewallPolicyResource=firewall_policy))
        return (self._client.firewallPolicies, 'Patch', self._messages.ComputeFirewallPoliciesPatchRequest(firewallPolicy=self.ref.Name(), firewallPolicyResource=firewall_policy))

    def _MakeDescribeRequestTuple(self, fp_id=None):
        if fp_id:
            return (self._client.firewallPolicies, 'Get', self._messages.ComputeFirewallPoliciesGetRequest(firewallPolicy=fp_id))
        return (self._client.firewallPolicies, 'Get', self._messages.ComputeFirewallPoliciesGetRequest(firewallPolicy=self.ref.Name()))

    def _MakeMoveRequestTuple(self, fp_id=None, parent_id=None):
        return (self._client.firewallPolicies, 'Move', self._messages.ComputeFirewallPoliciesMoveRequest(firewallPolicy=fp_id, parentId=parent_id))

    def _MakeCloneRulesRequestTuple(self, dest_fp_id=None, source_firewall_policy=None):
        return (self._client.firewallPolicies, 'CloneRules', self._messages.ComputeFirewallPoliciesCloneRulesRequest(firewallPolicy=dest_fp_id, sourceFirewallPolicy=source_firewall_policy))

    def _MakeListRequestTuple(self, parent_id):
        return (self._client.firewallPolicies, 'List', self._messages.ComputeFirewallPoliciesListRequest(parentId=parent_id))

    def _MakeCreateRequestTuple(self, firewall_policy, parent_id):
        return (self._client.firewallPolicies, 'Insert', self._messages.ComputeFirewallPoliciesInsertRequest(parentId=parent_id, firewallPolicy=firewall_policy))

    def AddAssociation(self, association=None, firewall_policy_id=None, replace_existing_association=False, batch_mode=False, only_generate_request=False):
        """Sends request to add an association."""
        if batch_mode:
            requests = [self._MakeAddAssociationRequestTuple(association, firewall_policy_id, replace_existing_association)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.AddAssociation(self._MakeAddAssociationRequestTuple(association, firewall_policy_id, replace_existing_association)[2])
        return self.WaitOperation(op_res, message='Adding an association for the organization firewall policy.')

    def DeleteAssociation(self, firewall_policy_id=None, batch_mode=False, only_generate_request=False):
        """Sends request to delete an association."""
        if batch_mode:
            requests = [self._MakeDeleteAssociationRequestTuple(firewall_policy_id)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.RemoveAssociation(self._MakeDeleteAssociationRequestTuple(firewall_policy_id)[2])
        return self.WaitOperation(op_res, message='Deleting the association for the organization firewall policy.')

    def ListAssociations(self, target_resource=None, batch_mode=False, only_generate_request=False):
        """Sends request to list all the associations."""
        if batch_mode:
            requests = [self._MakeListAssociationsRequestTuple(target_resource)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        return [self._service.ListAssociations(self._MakeListAssociationsRequestTuple(target_resource)[2])]

    def Delete(self, fp_id=None, batch_mode=False, only_generate_request=False):
        """Sends request to delete an organization firewall policy."""
        if batch_mode:
            requests = [self._MakeDeleteRequestTuple(fp_id=fp_id)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.Delete(self._MakeDeleteRequestTuple(fp_id=fp_id)[2])
        operation_poller = DeletePoller(self._service, self.ref)
        return self.WaitOperation(op_res, operation_poller=operation_poller, message='Deleting the organization firewall policy.')

    def WaitOperation(self, operation, operation_poller=None, message=None):
        if not operation_poller:
            operation_poller = poller.Poller(self._service, self.ref, has_project=self._api_has_project)
        if self._op_has_project and 'projects' not in operation.selfLink:
            operation.selfLink = operation.selfLink.replace('locations', 'projects/locations')
        operation_ref = self._resources.Parse(operation.selfLink, collection=OP_COLLECTION_NAME)
        return waiter.WaitFor(operation_poller, operation_ref, message)

    def Update(self, fp_id=None, only_generate_request=False, firewall_policy=None, batch_mode=False):
        """Sends request to update an organization firewall policy."""
        if batch_mode:
            requests = [self._MakeUpdateRequestTuple(fp_id=fp_id, firewall_policy=firewall_policy)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.Patch(self._MakeUpdateRequestTuple(fp_id=fp_id, firewall_policy=firewall_policy)[2])
        return self.WaitOperation(op_res, message='Updating the organization firewall policy.')

    def Move(self, only_generate_request=False, fp_id=None, parent_id=None, batch_mode=False):
        """Sends request to move the firewall policy to anther parent."""
        if batch_mode:
            requests = [self._MakeMoveRequestTuple(fp_id=fp_id, parent_id=parent_id)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.Move(self._MakeMoveRequestTuple(fp_id=fp_id, parent_id=parent_id)[2])
        return self.WaitOperation(op_res, message='Moving the organization firewall policy.')

    def CloneRules(self, only_generate_request=False, dest_fp_id=None, source_firewall_policy=None, batch_mode=False):
        """Sends request to clone all the rules from another firewall policy."""
        if batch_mode:
            requests = [self._MakeCloneRulesRequestTuple(dest_fp_id=dest_fp_id, source_firewall_policy=source_firewall_policy)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.CloneRules(self._MakeCloneRulesRequestTuple(dest_fp_id=dest_fp_id, source_firewall_policy=source_firewall_policy)[2])
        return self.WaitOperation(op_res, message='Cloning rules to the organization firewall policy.')

    def Describe(self, fp_id=None, batch_mode=False, only_generate_request=False):
        """Sends request to describe a firewall policy."""
        if batch_mode:
            requests = [self._MakeDescribeRequestTuple(fp_id=fp_id)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        return [self._service.Get(self._MakeDescribeRequestTuple(fp_id=fp_id)[2])]

    def List(self, parent_id=None, batch_mode=False, only_generate_request=False):
        """Sends request to list all the firewall policies."""
        if batch_mode:
            requests = [self._MakeListRequestTuple(parent_id)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        return [self._service.List(self._MakeListRequestTuple(parent_id)[2])]

    def Create(self, firewall_policy=None, parent_id=None, batch_mode=False, only_generate_request=False):
        """Sends request to create a firewall policy."""
        if batch_mode:
            requests = [self._MakeCreateRequestTuple(firewall_policy, parent_id)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.Insert(self._MakeCreateRequestTuple(firewall_policy, parent_id)[2])
        return self.WaitOperation(op_res, message='Creating the organization firewall policy.')