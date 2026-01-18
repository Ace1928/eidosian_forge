from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PatchJobInstanceDetails(_messages.Message):
    """Patch details for a VM instance. For more information about reviewing VM
  instance details, see [Listing all VM instance details for a specific patch
  job](https://cloud.google.com/compute/docs/os-patch-management/manage-patch-
  jobs#list-instance-details).

  Enums:
    StateValueValuesEnum: Current state of instance patch.

  Fields:
    attemptCount: The number of times the agent that the agent attempts to
      apply the patch.
    failureReason: If the patch fails, this field provides the reason.
    instanceSystemId: The unique identifier for the instance. This identifier
      is defined by the server.
    name: The instance name in the form `projects/*/zones/*/instances/*`
    state: Current state of instance patch.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Current state of instance patch.

    Values:
      PATCH_STATE_UNSPECIFIED: Unspecified.
      PENDING: The instance is not yet notified.
      INACTIVE: Instance is inactive and cannot be patched.
      NOTIFIED: The instance is notified that it should be patched.
      STARTED: The instance has started the patching process.
      DOWNLOADING_PATCHES: The instance is downloading patches.
      APPLYING_PATCHES: The instance is applying patches.
      REBOOTING: The instance is rebooting.
      SUCCEEDED: The instance has completed applying patches.
      SUCCEEDED_REBOOT_REQUIRED: The instance has completed applying patches
        but a reboot is required.
      FAILED: The instance has failed to apply the patch.
      ACKED: The instance acked the notification and will start shortly.
      TIMED_OUT: The instance exceeded the time out while applying the patch.
      RUNNING_PRE_PATCH_STEP: The instance is running the pre-patch step.
      RUNNING_POST_PATCH_STEP: The instance is running the post-patch step.
      NO_AGENT_DETECTED: The service could not detect the presence of the
        agent. Check to ensure that the agent is installed, running, and able
        to communicate with the service.
    """
        PATCH_STATE_UNSPECIFIED = 0
        PENDING = 1
        INACTIVE = 2
        NOTIFIED = 3
        STARTED = 4
        DOWNLOADING_PATCHES = 5
        APPLYING_PATCHES = 6
        REBOOTING = 7
        SUCCEEDED = 8
        SUCCEEDED_REBOOT_REQUIRED = 9
        FAILED = 10
        ACKED = 11
        TIMED_OUT = 12
        RUNNING_PRE_PATCH_STEP = 13
        RUNNING_POST_PATCH_STEP = 14
        NO_AGENT_DETECTED = 15
    attemptCount = _messages.IntegerField(1)
    failureReason = _messages.StringField(2)
    instanceSystemId = _messages.StringField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)