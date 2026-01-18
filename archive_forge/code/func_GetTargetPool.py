from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.target_pools import flags
def GetTargetPool(self, client, target_pool_ref):
    """Fetches the target pool resource."""
    objects = client.MakeRequests([(client.apitools_client.targetPools, 'Get', client.messages.ComputeTargetPoolsGetRequest(project=target_pool_ref.project, region=target_pool_ref.region, targetPool=target_pool_ref.Name()))])
    return objects[0]