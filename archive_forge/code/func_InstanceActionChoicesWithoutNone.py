from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def InstanceActionChoicesWithoutNone(flag_prefix=''):
    """Return possible instance action choices without NONE value."""
    return collections.OrderedDict([('refresh', 'Apply the new configuration without stopping VMs, if possible. For example, use ``refresh`` to apply changes that only affect metadata or additional disks.'), ('restart', 'Apply the new configuration without replacing VMs, if possible. For example, stopping VMs and starting them again is sufficient to apply changes to machine type.'), ('replace', 'Replace old VMs according to the --{flag_prefix}replacement-method flag.'.format(flag_prefix=flag_prefix))])