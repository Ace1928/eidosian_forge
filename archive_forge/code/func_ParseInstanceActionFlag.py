from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def ParseInstanceActionFlag(flag_name, instance_action_str, instance_action_enum):
    """Retrieves value of the instance action type.

  Args:
    flag_name: name of the flag associated with the parsed string.
    instance_action_str: string containing instance action value.
    instance_action_enum: enum type representing instance action values.

  Returns:
    InstanceAction enum object.
  """
    instance_actions_enum_map = {'none': instance_action_enum.NONE, 'refresh': instance_action_enum.REFRESH, 'restart': instance_action_enum.RESTART, 'replace': instance_action_enum.REPLACE}
    if instance_action_str not in instance_actions_enum_map:
        raise exceptions.InvalidArgumentException(flag_name, 'unknown instance action: ' + six.text_type(instance_action_str))
    return instance_actions_enum_map[instance_action_str]