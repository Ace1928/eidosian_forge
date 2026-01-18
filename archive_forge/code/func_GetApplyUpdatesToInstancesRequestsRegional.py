from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.instance_groups.flags import AutoDeleteFlag
from googlecloudsdk.command_lib.compute.instance_groups.flags import STATEFUL_IP_DEFAULT_INTERFACE_NAME
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_disk_getter
import six
def GetApplyUpdatesToInstancesRequestsRegional(holder, igm_ref, instances, minimal_action):
    """Immediately applies updates to instances (regional case)."""
    messages = holder.client.messages
    request = messages.RegionInstanceGroupManagersApplyUpdatesRequest(instances=instances, minimalAction=minimal_action, mostDisruptiveAllowedAction=messages.RegionInstanceGroupManagersApplyUpdatesRequest.MostDisruptiveAllowedActionValueValuesEnum.REPLACE)
    return messages.ComputeRegionInstanceGroupManagersApplyUpdatesToInstancesRequest(instanceGroupManager=igm_ref.Name(), regionInstanceGroupManagersApplyUpdatesRequest=request, project=igm_ref.project, region=igm_ref.region)