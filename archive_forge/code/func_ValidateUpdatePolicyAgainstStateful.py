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
def ValidateUpdatePolicyAgainstStateful(update_policy, group_ref, stateful_policy, client):
    """Validates and fixed update policy for stateful MIG.

  Sets default values in update_policy for stateful IGMs or throws exception
  if the wrong value is set explicitly.

  Args:
    update_policy: Update policy to be validated
    group_ref: Reference of IGM being validated
    stateful_policy: Stateful policy to check if the group is stateful
    client: The compute API client
  """
    if stateful_policy is None or _IsZonalGroup(group_ref):
        return
    redistribution_type_none = client.messages.InstanceGroupManagerUpdatePolicy.InstanceRedistributionTypeValueValuesEnum.NONE
    if not update_policy or update_policy.instanceRedistributionType != redistribution_type_none:
        raise exceptions.RequiredArgumentException('--instance-redistribution-type', "Stateful regional IGMs need to have instance redistribution type set to 'NONE'. Use '--instance-redistribution-type=NONE'.")