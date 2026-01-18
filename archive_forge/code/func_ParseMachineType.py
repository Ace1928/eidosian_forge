from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseMachineType(machine_type, fallback_zone=None):
    """Parses a machine type name using configuration properties for fallback.

  Args:
    machine_type: str, the machine type's ID, fully-qualified URL, or relative
        name
    fallback_zone: str, the zone to use if `machine_type` does not contain zone
        information. If None, and `machine_type` does not contain zone
        information, parsing will fail.

  Returns:
    googlecloudsdk.core.resources.Resource: a resource reference for the
    machine type
  """
    params = {'project': GetProject}
    if fallback_zone:
        params['zone'] = lambda z=fallback_zone: z
    return resources.REGISTRY.Parse(machine_type, params=params, collection='compute.machineTypes')