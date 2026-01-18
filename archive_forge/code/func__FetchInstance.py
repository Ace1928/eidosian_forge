from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _FetchInstance(self, args):
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    client = holder.client
    ssh_helper = ssh_utils.BaseSSHCLIHelper()
    instance_ref = flags.SSH_INSTANCE_RESOLVER.ResolveResources([args.instance_name], scope.ScopeEnum.ZONE, args.zone, holder.resources, scope_lister=flags.GetInstanceZoneScopeLister(client))[0]
    return (instance_ref, ssh_helper.GetInstance(client, instance_ref))