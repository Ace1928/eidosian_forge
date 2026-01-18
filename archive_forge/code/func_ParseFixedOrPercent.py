from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def ParseFixedOrPercent(flag_name, flag_param_name, fixed_or_percent_str, messages):
    """Retrieves value: number or percent.

  Args:
    flag_name: name of the flag associated with the parsed string.
    flag_param_name: name of the inner parameter of the flag.
    fixed_or_percent_str: string containing fixed or percent value.
    messages: module containing message classes.

  Returns:
    FixedOrPercent message object.
  """
    if fixed_or_percent_str is None:
        return None
    fixed = _ParseFixed(fixed_or_percent_str)
    if fixed is not None:
        return messages.FixedOrPercent(fixed=fixed)
    percent = _ParsePercent(fixed_or_percent_str)
    if percent is not None:
        if percent > 100:
            raise exceptions.InvalidArgumentException(flag_name, 'percentage cannot be higher than 100%.')
        return messages.FixedOrPercent(percent=percent)
    raise exceptions.InvalidArgumentException(flag_name, flag_param_name + ' has to be non-negative integer number or percent.')