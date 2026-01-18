from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.core.exceptions import Error
def IsProvisioningTypeIops(disk_type):
    """Check if the given disk type (name or URI) supports IOPS provisioning.

  Args:
    disk_type: name of URI of the disk type to be checked.

  Returns:
    Whether the disk_type supports IOPS provisioning.
  """
    return disk_type.endswith('/pd-extreme') or disk_type.endswith('/cs-extreme') or disk_type.endswith('/hyperdisk-extreme') or disk_type.endswith('/hyperdisk-balanced') or (disk_type in ['pd-extreme', 'cs-extreme', 'hyperdisk-extreme', 'hyperdisk-balanced'])