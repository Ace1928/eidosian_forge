from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceStatusValueValuesEnum(_messages.Enum):
    """[Output Only] The status of the instance. This field is empty when the
    instance does not exist.

    Values:
      DEPROVISIONING: The instance is halted and we are performing tear down
        tasks like network deprogramming, releasing quota, IP, tearing down
        disks etc.
      PROVISIONING: Resources are being allocated for the instance.
      REPAIRING: The instance is in repair.
      RUNNING: The instance is running.
      STAGING: All required resources have been allocated and the instance is
        being started.
      STOPPED: The instance has stopped successfully.
      STOPPING: The instance is currently stopping (either being deleted or
        killed).
      SUSPENDED: The instance has suspended.
      SUSPENDING: The instance is suspending.
      TERMINATED: The instance has stopped (either by explicit action or
        underlying failure).
    """
    DEPROVISIONING = 0
    PROVISIONING = 1
    REPAIRING = 2
    RUNNING = 3
    STAGING = 4
    STOPPED = 5
    STOPPING = 6
    SUSPENDED = 7
    SUSPENDING = 8
    TERMINATED = 9