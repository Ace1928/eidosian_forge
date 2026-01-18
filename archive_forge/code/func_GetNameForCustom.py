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
def GetNameForCustom(custom_cpu, custom_memory_mib, ext=False, vm_type=False):
    """Creates a custom machine type name from the desired CPU and memory specs.

  Args:
    custom_cpu: the number of cpu desired for the custom machine type
    custom_memory_mib: the amount of ram desired in MiB for the custom machine
      type instance
    ext: extended custom machine type should be used if true
    vm_type: VM instance generation

  Returns:
    The custom machine type name for the 'instance create' call
  """
    if vm_type:
        machine_type = '{0}-custom-{1}-{2}'.format(vm_type, custom_cpu, custom_memory_mib)
    else:
        machine_type = 'custom-{0}-{1}'.format(custom_cpu, custom_memory_mib)
    if ext:
        machine_type += '-ext'
    return machine_type