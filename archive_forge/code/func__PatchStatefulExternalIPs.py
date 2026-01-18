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
def _PatchStatefulExternalIPs(messages, per_instance_config, ips_to_update, ips_to_remove):
    """Patch and return the updated list of stateful external IPs."""
    existing_ips = per_instance_config.preservedState.externalIPs.additionalProperties if per_instance_config.preservedState.externalIPs else []
    ips_to_update_dict = {ip.get('interface-name', instance_groups_flags.STATEFUL_IP_DEFAULT_INTERFACE_NAME): ip for ip in iter(ips_to_update or [])}
    UpdateGA._VerifyStatefulIPsToRemoveSet('--remove-stateful-external-ips', existing_ips, ips_to_remove)
    new_stateful_ips, remaining_ips_to_update = UpdateGA._UpdateExistingIPs(messages, existing_ips, ips_to_update_dict, ips_to_remove)
    new_stateful_ips.extend(UpdateGA._CreateExternalIPs(messages, remaining_ips_to_update))
    per_instance_config.preservedState.externalIPs = messages.PreservedState.ExternalIPsValue(additionalProperties=new_stateful_ips)