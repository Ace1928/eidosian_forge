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
def CallPerInstanceConfigUpdate(holder, igm_ref, per_instance_config_message):
    """Calls proper (zonal or regional) resource for instance config update."""
    messages = holder.client.messages
    if igm_ref.Collection() == 'compute.instanceGroupManagers':
        service = holder.client.apitools_client.instanceGroupManagers
        request = messages.ComputeInstanceGroupManagersUpdatePerInstanceConfigsRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagersUpdatePerInstanceConfigsReq=messages.InstanceGroupManagersUpdatePerInstanceConfigsReq(perInstanceConfigs=[per_instance_config_message]), project=igm_ref.project, zone=igm_ref.zone)
        operation_collection = 'compute.zoneOperations'
    elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
        service = holder.client.apitools_client.regionInstanceGroupManagers
        request = messages.ComputeRegionInstanceGroupManagersUpdatePerInstanceConfigsRequest(instanceGroupManager=igm_ref.Name(), regionInstanceGroupManagerUpdateInstanceConfigReq=messages.RegionInstanceGroupManagerUpdateInstanceConfigReq(perInstanceConfigs=[per_instance_config_message]), project=igm_ref.project, region=igm_ref.region)
        operation_collection = 'compute.regionOperations'
    else:
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
    operation = service.UpdatePerInstanceConfigs(request)
    operation_ref = holder.resources.Parse(operation.selfLink, collection=operation_collection)
    return operation_ref