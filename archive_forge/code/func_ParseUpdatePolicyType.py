from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def ParseUpdatePolicyType(flag_name, policy_type_str, messages):
    """Retrieves value of update policy type: opportunistic or proactive.

  Args:
    flag_name: name of the flag associated with the parsed string.
    policy_type_str: string containing update policy type.
    messages: module containing message classes.

  Returns:
    InstanceGroupManagerUpdatePolicy.TypeValueValuesEnum message enum value.
  """
    if policy_type_str == 'opportunistic':
        return messages.InstanceGroupManagerUpdatePolicy.TypeValueValuesEnum.OPPORTUNISTIC
    elif policy_type_str == 'proactive':
        return messages.InstanceGroupManagerUpdatePolicy.TypeValueValuesEnum.PROACTIVE
    raise exceptions.InvalidArgumentException(flag_name, 'unknown update policy.')