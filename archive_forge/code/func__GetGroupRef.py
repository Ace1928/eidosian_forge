from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
def _GetGroupRef(self, args, resources, client):
    if args.instance_group:
        return flags.MULTISCOPE_INSTANCE_GROUP_ARG.ResolveAsResource(args, resources, scope_lister=compute_flags.GetDefaultScopeLister(client))
    if args.network_endpoint_group:
        return flags.GetNetworkEndpointGroupArg(support_global_neg=self.support_global_neg, support_region_neg=self.support_region_neg).ResolveAsResource(args, resources, scope_lister=compute_flags.GetDefaultScopeLister(client))