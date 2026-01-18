from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def _ParsePercent(fixed_or_percent_str):
    """Retrieves percent value from string."""
    if re.match('^\\d+%$', fixed_or_percent_str):
        percent = int(fixed_or_percent_str[:-1])
        return percent
    return None