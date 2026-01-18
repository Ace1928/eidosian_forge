from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.reservations import util as reservation_util
from googlecloudsdk.core.util import times
def MakeAllocatedInstanceProperties(messages, machine_type, min_cpu_platform, local_ssds, accelerators, location_hint=None, freeze_duration=None, freeze_interval=None):
    """Constructs an instance propteries for reservations message object."""
    prop_msgs = messages.AllocationSpecificSKUAllocationReservedInstanceProperties
    instance_properties = prop_msgs(machineType=machine_type, guestAccelerators=accelerators, minCpuPlatform=min_cpu_platform, localSsds=local_ssds)
    if location_hint:
        instance_properties.locationHint = location_hint
    if freeze_duration:
        instance_properties.maintenanceFreezeDurationHours = freeze_duration // 3600
    if freeze_interval:
        instance_properties.maintenanceInterval = messages.AllocationSpecificSKUAllocationReservedInstanceProperties.MaintenanceIntervalValueValuesEnum(freeze_interval)
    return instance_properties