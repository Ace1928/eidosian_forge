from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
def GetRoutePolicy(self, client, router_ref, policy_name):
    request = (client.apitools_client.routers, 'GetRoutePolicy', client.messages.ComputeRoutersGetRoutePolicyRequest(**router_ref.AsDict(), policy=policy_name))
    return client.MakeRequests([request])[0]