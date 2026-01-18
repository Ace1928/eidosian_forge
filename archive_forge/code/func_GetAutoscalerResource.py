from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
def GetAutoscalerResource(self, client, resources, igm_ref, args):
    if _IsZonalGroup(igm_ref):
        scope_type = 'zone'
        location = managed_instance_groups_utils.CreateZoneRef(resources, igm_ref)
        zones, regions = ([location], None)
    else:
        scope_type = 'region'
        location = managed_instance_groups_utils.CreateRegionRef(resources, igm_ref)
        zones, regions = (None, [location])
    autoscaler = managed_instance_groups_utils.AutoscalerForMig(mig_name=args.name, autoscalers=managed_instance_groups_utils.AutoscalersForLocations(regions=regions, zones=zones, client=client), location=location, scope_type=scope_type)
    if autoscaler is None:
        raise managed_instance_groups_utils.ResourceNotFoundException('The managed instance group is not autoscaled.')
    return autoscaler