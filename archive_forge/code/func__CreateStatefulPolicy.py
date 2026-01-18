from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import stateful_policy_utils as policy_utils
from googlecloudsdk.api_lib.compute.managed_instance_groups_utils import ValueOrNone
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as managed_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def _CreateStatefulPolicy(self, args, client):
    """Create stateful policy from disks of args --stateful-disk, and ips of args --stateful-external-ips and --stateful-internal-ips."""
    stateful_disks = []
    for stateful_disk_dict in args.stateful_disk or []:
        stateful_disks.append(policy_utils.MakeStatefulPolicyPreservedStateDiskEntry(client.messages, stateful_disk_dict))
    stateful_disks.sort(key=lambda x: x.key)
    stateful_policy = policy_utils.MakeStatefulPolicy(client.messages, stateful_disks)
    stateful_internal_ips = []
    for stateful_ip_dict in args.stateful_internal_ip or []:
        stateful_internal_ips.append(policy_utils.MakeStatefulPolicyPreservedStateInternalIPEntry(client.messages, stateful_ip_dict))
    stateful_internal_ips.sort(key=lambda x: x.key)
    stateful_policy.preservedState.internalIPs = client.messages.StatefulPolicyPreservedState.InternalIPsValue(additionalProperties=stateful_internal_ips)
    stateful_external_ips = []
    for stateful_ip_dict in args.stateful_external_ip or []:
        stateful_external_ips.append(policy_utils.MakeStatefulPolicyPreservedStateExternalIPEntry(client.messages, stateful_ip_dict))
    stateful_external_ips.sort(key=lambda x: x.key)
    stateful_policy.preservedState.externalIPs = client.messages.StatefulPolicyPreservedState.ExternalIPsValue(additionalProperties=stateful_external_ips)
    return stateful_policy