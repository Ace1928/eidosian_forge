from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_getter
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_messages
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_disk_getter
import six
@staticmethod
def _UpdateExistingIPs(messages, existing_ips, ips_to_update_dict, ips_to_remove):
    new_stateful_ips = []
    remaining_ips_to_update = dict(ips_to_update_dict)
    ips_to_remove_set = set(ips_to_remove or [])
    for current_stateful_ip in existing_ips:
        interface_name = current_stateful_ip.key
        if interface_name in ips_to_remove_set:
            continue
        if interface_name in remaining_ips_to_update:
            instance_configs_messages.PatchPreservedStateNetworkIpEntry(messages, current_stateful_ip.value, remaining_ips_to_update[interface_name])
            del remaining_ips_to_update[interface_name]
        new_stateful_ips.append(current_stateful_ip)
    return (new_stateful_ips, remaining_ips_to_update)