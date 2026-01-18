from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerformanceProvisioningTypeValueValuesEnum(_messages.Enum):
    """Provisioning type of the performance-related parameters of the pool,
    such as throughput and IOPS.

    Values:
      ADVANCED: Advanced provisioning "thinly" allocates the related resource.
      STANDARD: Standard provisioning allocates the related resource for the
        pool disks' exclusive use.
      UNSPECIFIED: <no description>
    """
    ADVANCED = 0
    STANDARD = 1
    UNSPECIFIED = 2