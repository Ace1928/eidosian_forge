from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.core.exceptions import Error
def IsProvisioningTypeThroughput(disk_type):
    """Check if the given disk type (name or URI) supports throughput provisioning.

  Args:
    disk_type: name of URI of the disk type to be checked.

  Returns:
    Boolean, true if the disk_type supports throughput provisioning, false
    otherwise.
  """
    return disk_type.endswith('/cs-throughput') or disk_type.endswith('/hyperdisk-throughput') or disk_type.endswith('/hyperdisk-balanced') or disk_type.endswith('/hyperdisk-ml') or (disk_type in _THROUGHPUT_PROVISIONING_SUPPORTED_DISK_TYPES)