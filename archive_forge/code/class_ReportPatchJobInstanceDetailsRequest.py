from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ReportPatchJobInstanceDetailsRequest(_messages.Message):
    """Request to report the patch status for an instance.

  Enums:
    StateValueValuesEnum: State of current patch execution on the instance.

  Fields:
    attemptCount: Number of times the agent attempted to apply the patch.
    failureReason: Reason for failure.
    instanceIdToken: This is the GCE instance identity token described in
      https://cloud.google.com/compute/docs/instances/verifying-instance-
      identity where the audience is 'osconfig.googleapis.com' and the format
      is 'full'.
    instanceSystemId: Required. The unique, system-generated identifier for
      the instance.  This is the unchangeable, auto-generated ID assigned to
      the instance upon creation. This is needed here because GCE instance
      names are not tombstoned; it is possible to delete an instance and
      create a new one with the same name; this provides a mechanism for this
      API to identify distinct instances in this case.
    patchJob: Unique identifier of the patch job this request applies to.
    state: State of current patch execution on the instance.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of current patch execution on the instance.

    Values:
      PATCH_STATE_UNSPECIFIED: Unspecified.
      PENDING: The instance has not been notified yet.
      INACTIVE: Instance is inactive and cannot be patched.
      NOTIFIED: The instance has been notified that it should patch.
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
    attemptCount = _messages.IntegerField(1)
    failureReason = _messages.StringField(2)
    instanceIdToken = _messages.StringField(3)
    instanceSystemId = _messages.StringField(4)
    patchJob = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)