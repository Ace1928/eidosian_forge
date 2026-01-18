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
def GetCpuRamVmFamilyFromCustomName(name):
    """Gets the CPU and memory specs from the custom machine type name.

  Args:
    name: the custom machine type name for the 'instance create' call

  Returns:
    A three-tuple with the vm family, number of cpu and amount of memory for the
    custom machine type.
    custom_family, the name of the VM family
    custom_cpu, the number of cpu desired for the custom machine type instance
    custom_memory_mib, the amount of ram desired in MiB for the custom machine
      type instance
    None for both variables otherwise
  """
    check_custom = re.search('([a-zA-Z0-9]+)-custom-([0-9]+)-([0-9]+)', name)
    if check_custom:
        custom_family = check_custom.group(1)
        custom_cpu = check_custom.group(2)
        custom_memory_mib = check_custom.group(3)
        return (custom_family, custom_cpu, custom_memory_mib)
    return (None, None, None)