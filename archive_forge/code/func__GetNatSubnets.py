from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnetwork_flags
from googlecloudsdk.command_lib.compute.service_attachments import flags
from googlecloudsdk.command_lib.compute.service_attachments import service_attachments_utils
def _GetNatSubnets(self, holder, args):
    """Returns nat subnetwork urls from the argument."""
    nat_subnetwork_refs = self.NAT_SUBNETWORK_ARG.ResolveAsResource(args, holder.resources, default_scope=compute_scope.ScopeEnum.REGION, scope_lister=compute_flags.GetDefaultScopeLister(holder.client))
    nat_subnetworks = [nat_subnetwork_ref.SelfLink() for nat_subnetwork_ref in nat_subnetwork_refs]
    return nat_subnetworks