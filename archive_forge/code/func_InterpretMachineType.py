from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def InterpretMachineType(machine_type, custom_cpu, custom_memory, ext=True, vm_type=False, confidential_vm_type=None):
    """Interprets the machine type for the instance.

  Args:
    machine_type: name of existing machine type, eg. n1-standard
    custom_cpu: number of CPU cores for custom machine type,
    custom_memory: amount of RAM memory in bytes for custom machine type,
    ext: extended custom machine type should be used if true,
    vm_type:  VM instance generation
    confidential_vm_type: If not None, use default machine type based on
        confidential-VM encryption type.

  Returns:
    A string representing the URL naming a machine-type.

  Raises:
    calliope_exceptions.RequiredArgumentException when only one of the two
      custom machine type flags are used.
    calliope_exceptions.InvalidArgumentException when both the machine type and
      custom machine type flags are used to generate a new instance.
  """
    if machine_type:
        machine_type_name = machine_type
    elif confidential_vm_type is not None:
        machine_type_name = constants.DEFAULT_MACHINE_TYPE_FOR_CONFIDENTIAL_VMS[confidential_vm_type]
    else:
        machine_type_name = constants.DEFAULT_MACHINE_TYPE
    if custom_cpu or custom_memory or ext:
        if not custom_cpu:
            raise calliope_exceptions.RequiredArgumentException('--custom-cpu', 'Both [--custom-cpu] and [--custom-memory] must be set to create a custom machine type instance.')
        if not custom_memory:
            raise calliope_exceptions.RequiredArgumentException('--custom-memory', 'Both [--custom-cpu] and [--custom-memory] must be set to create a custom machine type instance.')
        if machine_type:
            raise calliope_exceptions.InvalidArgumentException('--machine-type', 'Cannot set both [--machine-type] and [--custom-cpu]/[--custom-memory] for the same instance.')
        custom_type_string = GetNameForCustom(custom_cpu, custom_memory // 2 ** 20, ext, vm_type)
        machine_type_name = custom_type_string
    return machine_type_name