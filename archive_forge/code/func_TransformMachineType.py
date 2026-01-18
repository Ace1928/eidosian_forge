from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformMachineType(r, undefined=''):
    """Return the formatted name for a machine type.

  Args:
    r: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    The formatted name for a machine type.
  """
    if not isinstance(r, six.string_types):
        return undefined
    custom_family, custom_cpu, custom_ram = instance_utils.GetCpuRamVmFamilyFromCustomName(r)
    if not custom_family or not custom_cpu or (not custom_ram):
        return r
    custom_ram_gb = '{0:.2f}'.format(float(custom_ram) / 2 ** 10)
    return 'custom ({0}, {1} vCPU, {2} GiB)'.format(custom_family, custom_cpu, custom_ram_gb)