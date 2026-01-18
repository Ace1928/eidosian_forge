from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class RegionNetworkFirewallPolicy(object):
    """Abstracts a region network firewall policy resource."""

    def __init__(self, ref, compute_client=None):
        self.ref = ref
        self._compute_client = compute_client

    @property
    def _client(self):
        return self._compute_client.apitools_client

    @property
    def _messages(self):
        return self._compute_client.messages

    @property
    def _service(self):
        return self._client.regionNetworkFirewallPolicies

    def _HasProject(self, collection):
        collection_info = self._resources.GetCollectionInfo(collection, self._version)
        return 'projects' in collection_info.path or 'projects' in collection_info.base_url

    def _MakeAddAssociationRequestTuple(self, association, firewall_policy, replace_existing_association):
        return (self._client.regionNetworkFirewallPolicies, 'AddAssociation', self._messages.ComputeRegionNetworkFirewallPoliciesAddAssociationRequest(firewallPolicyAssociation=association, firewallPolicy=firewall_policy, replaceExistingAssociation=replace_existing_association, project=self.ref.project, region=self.ref.region))

    def _MakePatchAssociationRequestTuple(self, association, firewall_policy):
        return (self._client.regionNetworkFirewallPolicies, 'PatchAssociation', self._messages.ComputeRegionNetworkFirewallPoliciesPatchAssociationRequest(firewallPolicyAssociation=association, firewallPolicy=firewall_policy, project=self.ref.project, region=self.ref.region))

    def _MakeCloneRulesRequestTuple(self, source_firewall_policy):
        return (self._client.regionNetworkFirewallPolicies, 'CloneRules', self._messages.ComputeRegionNetworkFirewallPoliciesCloneRulesRequest(firewallPolicy=self.ref.Name(), sourceFirewallPolicy=source_firewall_policy, project=self.ref.project, region=self.ref.region))

    def _MakeCreateRequestTuple(self, firewall_policy):
        return (self._client.regionNetworkFirewallPolicies, 'Insert', self._messages.ComputeRegionNetworkFirewallPoliciesInsertRequest(firewallPolicy=firewall_policy, project=self.ref.project, region=self.ref.region))

    def _MakeDeleteRequestTuple(self, firewall_policy):
        return (self._client.regionNetworkFirewallPolicies, 'Delete', self._messages.ComputeRegionNetworkFirewallPoliciesDeleteRequest(firewallPolicy=firewall_policy, project=self.ref.project, region=self.ref.region))

    def _MakeDescribeRequestTuple(self):
        return (self._client.regionNetworkFirewallPolicies, 'Get', self._messages.ComputeRegionNetworkFirewallPoliciesGetRequest(firewallPolicy=self.ref.Name(), project=self.ref.project, region=self.ref.region))

    def _MakeDeleteAssociationRequestTuple(self, firewall_policy, name):
        return (self._client.regionNetworkFirewallPolicies, 'RemoveAssociation', self._messages.ComputeRegionNetworkFirewallPoliciesRemoveAssociationRequest(firewallPolicy=firewall_policy, name=name, project=self.ref.project, region=self.ref.region))

    def _MakeUpdateRequestTuple(self, firewall_policy=None):
        return (self._client.regionNetworkFirewallPolicies, 'Patch', self._messages.ComputeRegionNetworkFirewallPoliciesPatchRequest(firewallPolicy=self.ref.Name(), firewallPolicyResource=firewall_policy, project=self.ref.project, region=self.ref.region))

    def CloneRules(self, source_firewall_policy=None, only_generate_request=False):
        """Sends request to clone all the rules from another firewall policy."""
        requests = [self._MakeCloneRulesRequestTuple(source_firewall_policy=source_firewall_policy)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests

    def Create(self, firewall_policy=None, only_generate_request=False):
        """Sends request to create a region network firewall policy."""
        requests = [self._MakeCreateRequestTuple(firewall_policy=firewall_policy)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests

    def Delete(self, firewall_policy=None, only_generate_request=False):
        """Sends request to delete a region network firewall policy."""
        requests = [self._MakeDeleteRequestTuple(firewall_policy=firewall_policy)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests

    def Describe(self, only_generate_request=False):
        """Sends request to describe a region network firewall policy."""
        requests = [self._MakeDescribeRequestTuple()]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests

    def Update(self, firewall_policy=None, only_generate_request=False):
        """Sends request to update a region network firewall policy."""
        requests = [self._MakeUpdateRequestTuple(firewall_policy)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests

    def AddAssociation(self, association=None, firewall_policy=None, replace_existing_association=False, only_generate_request=False):
        """Sends request to add an association."""
        requests = [self._MakeAddAssociationRequestTuple(association, firewall_policy, replace_existing_association)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests

    def PatchAssociation(self, association=None, firewall_policy=None, only_generate_request=False):
        """Sends request to patch an association."""
        requests = [self._MakePatchAssociationRequestTuple(association, firewall_policy)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests

    def DeleteAssociation(self, firewall_policy=None, name=None, only_generate_request=False):
        """Sends request to delete an association."""
        requests = [self._MakeDeleteAssociationRequestTuple(firewall_policy, name)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests