from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
def _MakePatchRequestTuple(self, description, admin_enabled, bandwidth, partner_metadata, mtu=None, stack_type=None, candidate_ipv6_subnets=None, cloud_router_ipv6_interface_id=None, customer_router_ipv6_interface_id=None, labels=None, label_fingerprint=None, multicast_enabled=None):
    """Make an interconnect attachment patch request."""
    interconnect_attachment = self._messages.InterconnectAttachment(name=self.ref.Name(), description=description, adminEnabled=admin_enabled, bandwidth=bandwidth, partnerMetadata=partner_metadata)
    if mtu is not None:
        interconnect_attachment.mtu = mtu
    if stack_type is not None:
        interconnect_attachment.stackType = self._messages.InterconnectAttachment.StackTypeValueValuesEnum(stack_type)
    if labels is not None:
        interconnect_attachment.labels = labels
    if label_fingerprint is not None:
        interconnect_attachment.labelFingerprint = label_fingerprint
    if candidate_ipv6_subnets is not None:
        interconnect_attachment.candidateIpv6Subnets = candidate_ipv6_subnets
    if cloud_router_ipv6_interface_id is not None:
        interconnect_attachment.cloudRouterIpv6InterfaceId = cloud_router_ipv6_interface_id
    if customer_router_ipv6_interface_id is not None:
        interconnect_attachment.customerRouterIpv6InterfaceId = customer_router_ipv6_interface_id
    if multicast_enabled is not None:
        interconnect_attachment.multicastEnabled = multicast_enabled
    return (self._client.interconnectAttachments, 'Patch', self._messages.ComputeInterconnectAttachmentsPatchRequest(project=self.ref.project, region=self.ref.region, interconnectAttachment=self.ref.Name(), interconnectAttachmentResource=interconnect_attachment))