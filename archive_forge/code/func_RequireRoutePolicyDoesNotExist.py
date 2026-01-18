from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def RequireRoutePolicyDoesNotExist(self, client, router_ref, policy_name):
    request = (client.apitools_client.routers, 'GetRoutePolicy', client.messages.ComputeRoutersGetRoutePolicyRequest(**router_ref.AsDict(), policy=policy_name))
    try:
        client.MakeRequests([request])
    except Exception as exception:
        if "Could not fetch resource:\n - Invalid value for field 'policy': " in exception.__str__():
            return
        raise
    raise exceptions.BadArgumentException('policy-name', "A policy named '{0}' already exists".format(policy_name))