from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TerminationReasonValueValuesEnum(_messages.Enum):
    """Reason for termination

    Values:
      BAD_BILLING_ACCOUNT: Terminated due to bad billing
      CLOUD_ABUSE_DETECTED: Terminated by Cloud Abuse team
      DISK_ERROR: Terminated due to disk errors
      FREE_TRIAL_EXPIRED: Terminated due to free trial expired
      INSTANCE_UPDATE_REQUIRED_RESTART: Instance.update initiated which
        required RESTART
      INTERNAL_ERROR: Terminated due to internal error
      KMS_REJECTION: Terminated due to Key Management Service (KMS) key
        failure.
      MANAGED_INSTANCE_GROUP: Terminated by managed instance group
      OS_TERMINATED: Terminated from the OS level
      PREEMPTED: Terminated due to preemption
      SCHEDULED_STOP: Terminated due to scheduled stop
      SHUTDOWN_DUE_TO_MAINTENANCE: Terminated due to maintenance
      USER_TERMINATED: Terminated by user
    """
    BAD_BILLING_ACCOUNT = 0
    CLOUD_ABUSE_DETECTED = 1
    DISK_ERROR = 2
    FREE_TRIAL_EXPIRED = 3
    INSTANCE_UPDATE_REQUIRED_RESTART = 4
    INTERNAL_ERROR = 5
    KMS_REJECTION = 6
    MANAGED_INSTANCE_GROUP = 7
    OS_TERMINATED = 8
    PREEMPTED = 9
    SCHEDULED_STOP = 10
    SHUTDOWN_DUE_TO_MAINTENANCE = 11
    USER_TERMINATED = 12