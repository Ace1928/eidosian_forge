from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2JobTrigger(_messages.Message):
    """Contains a configuration to make api calls on a repeating basis. See
  https://cloud.google.com/sensitive-data-protection/docs/concepts-job-
  triggers to learn more.

  Enums:
    StatusValueValuesEnum: Required. A status for this trigger.

  Fields:
    createTime: Output only. The creation timestamp of a triggeredJob.
    description: User provided description (max 256 chars)
    displayName: Display name (max 100 chars)
    errors: Output only. A stream of errors encountered when the trigger was
      activated. Repeated errors may result in the JobTrigger automatically
      being paused. Will return the last 100 errors. Whenever the JobTrigger
      is modified this list will be cleared.
    inspectJob: For inspect jobs, a snapshot of the configuration.
    lastRunTime: Output only. The timestamp of the last time this trigger
      executed.
    name: Unique resource name for the triggeredJob, assigned by the service
      when the triggeredJob is created, for example `projects/dlp-test-
      project/jobTriggers/53234423`.
    status: Required. A status for this trigger.
    triggers: A list of triggers which will be OR'ed together. Only one in the
      list needs to trigger for a job to be started. The list may contain only
      a single Schedule trigger and must have at least one object.
    updateTime: Output only. The last update timestamp of a triggeredJob.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """Required. A status for this trigger.

    Values:
      STATUS_UNSPECIFIED: Unused.
      HEALTHY: Trigger is healthy.
      PAUSED: Trigger is temporarily paused.
      CANCELLED: Trigger is cancelled and can not be resumed.
    """
        STATUS_UNSPECIFIED = 0
        HEALTHY = 1
        PAUSED = 2
        CANCELLED = 3
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    errors = _messages.MessageField('GooglePrivacyDlpV2Error', 4, repeated=True)
    inspectJob = _messages.MessageField('GooglePrivacyDlpV2InspectJobConfig', 5)
    lastRunTime = _messages.StringField(6)
    name = _messages.StringField(7)
    status = _messages.EnumField('StatusValueValuesEnum', 8)
    triggers = _messages.MessageField('GooglePrivacyDlpV2Trigger', 9, repeated=True)
    updateTime = _messages.StringField(10)