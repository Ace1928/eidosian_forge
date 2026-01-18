from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EkmProvisioningStateValueValuesEnum(_messages.Enum):
    """Indicates Ekm enrollment Provisioning of a given workload.

    Values:
      EKM_PROVISIONING_STATE_UNSPECIFIED: Default State for Ekm Provisioning
      EKM_PROVISIONING_STATE_PENDING: Pending State for Ekm Provisioning
      EKM_PROVISIONING_STATE_FAILED: Failed State for Ekm Provisioning
      EKM_PROVISIONING_STATE_COMPLETED: Completed State for Ekm Provisioning
    """
    EKM_PROVISIONING_STATE_UNSPECIFIED = 0
    EKM_PROVISIONING_STATE_PENDING = 1
    EKM_PROVISIONING_STATE_FAILED = 2
    EKM_PROVISIONING_STATE_COMPLETED = 3