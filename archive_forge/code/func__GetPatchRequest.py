from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.routers import flags as routers_flags
from googlecloudsdk.command_lib.compute.routers.nats import nats_utils
from googlecloudsdk.command_lib.compute.routers.nats.rules import flags as rules_flags
from googlecloudsdk.command_lib.compute.routers.nats.rules import rules_utils
def _GetPatchRequest(self, client, router_ref, router):
    return (client.apitools_client.routers, 'Patch', client.messages.ComputeRoutersPatchRequest(router=router_ref.Name(), routerResource=router, region=router_ref.region, project=router_ref.project))