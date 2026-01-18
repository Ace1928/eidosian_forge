from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.compute.reservations import util
from googlecloudsdk.core import yaml
def _ConvertYAMLToMessage(messages, reservations_yaml, resources):
    """Converts the fields in yaml to allocation message object."""
    if not reservations_yaml:
        return []
    allocations_msg = []
    for a in reservations_yaml:
        accelerators = util.MakeGuestAccelerators(messages, a.get('accelerator', None))
        local_ssds = util.MakeLocalSsds(messages, a.get('local_ssd', None))
        share_settings = util.MakeShareSettingsWithDict(messages, a, a.get('share_setting', None))
        resource_policies = util.MakeResourcePolicies(messages, a, a.get('resource_policies', None), resources)
        specific_allocation = util.MakeSpecificSKUReservationMessage(messages, a.get('vm_count', None), accelerators, local_ssds, a.get('machine_type', None), a.get('min_cpu_platform', None))
        a_msg = util.MakeReservationMessage(messages, a.get('reservation', None), share_settings, specific_allocation, resource_policies, a.get('require_specific_reservation', None), a.get('reservation_zone', None))
        allocations_msg.append(a_msg)
    return allocations_msg