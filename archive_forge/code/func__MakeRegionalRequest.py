from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.target_tcp_proxies import flags
def _MakeRegionalRequest(self, ref, holder):
    client = holder.client.apitools_client
    messages = holder.client.messages
    request = messages.ComputeRegionTargetTcpProxiesGetRequest(project=ref.project, targetTcpProxy=ref.Name(), region=ref.region)
    errors = []
    resources = holder.client.MakeRequests([(client.regionTargetTcpProxies, 'Get', request)], errors)
    if errors:
        utils.RaiseToolException(errors)
    return resources[0]