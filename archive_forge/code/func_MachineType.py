from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def MachineType(instance=None, tier=None, memory=None, cpu=None):
    """Generates the machine type for the instance.

  Adapted from compute.

  Args:
    instance: sql_messages.DatabaseInstance, The original instance, if it might
      be needed to generate the machine type.
    tier: string, the v1 or v2 tier.
    memory: string, the amount of memory.
    cpu: int, the number of CPUs.

  Returns:
    A string representing the URL naming a machine-type.

  Raises:
    exceptions.RequiredArgumentException when only one of the two custom
        machine type flags are used, or when none of the flags are used.
    exceptions.InvalidArgumentException when both the tier and
        custom machine type flags are used to generate a new instance.
  """
    machine_type = None
    if tier:
        machine_type = tier
    if cpu or memory:
        if not cpu:
            raise exceptions.RequiredArgumentException('--cpu', 'Both [--cpu] and [--memory] must be set to create a custom machine type instance.')
        if not memory:
            raise exceptions.RequiredArgumentException('--memory', 'Both [--cpu] and [--memory] must be set to create a custom machine type instance.')
        if tier:
            raise exceptions.InvalidArgumentException('--tier', 'Cannot set both [--tier] and [--cpu]/[--memory] for the same instance.')
        custom_type_string = _CustomMachineTypeString(cpu, memory // 2 ** 20)
        machine_type = custom_type_string
    if not machine_type and (not instance):
        machine_type = constants.DEFAULT_MACHINE_TYPE
    return machine_type