from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Schedule(_messages.Message):
    """An instance of a Schedule periodically schedules runs to make API calls
  based on user specified time specification and API request type.

  Enums:
    StateValueValuesEnum: Output only. The state of this Schedule.

  Fields:
    allowQueueing: Optional. Whether new scheduled runs can be queued when
      max_concurrent_runs limit is reached. If set to true, new runs will be
      queued instead of skipped. Default to false.
    catchUp: Output only. Whether to backfill missed runs when the schedule is
      resumed from PAUSED state. If set to true, all missed runs will be
      scheduled. New runs will be scheduled after the backfill is complete.
      Default to false.
    createPipelineJobRequest: Request for PipelineService.CreatePipelineJob.
      CreatePipelineJobRequest.parent field is required (format:
      projects/{project}/locations/{location}).
    createTime: Output only. Timestamp when this Schedule was created.
    cron: Cron schedule (https://en.wikipedia.org/wiki/Cron) to launch
      scheduled runs. To explicitly set a timezone to the cron tab, apply a
      prefix in the cron tab: "CRON_TZ=${IANA_TIME_ZONE}" or
      "TZ=${IANA_TIME_ZONE}". The ${IANA_TIME_ZONE} may only be a valid string
      from IANA time zone database. For example, "CRON_TZ=America/New_York 1 *
      * * *", or "TZ=America/New_York 1 * * * *".
    displayName: Required. User provided name of the Schedule. The name can be
      up to 128 characters long and can consist of any UTF-8 characters.
    endTime: Optional. Timestamp after which no new runs can be scheduled. If
      specified, The schedule will be completed when either end_time is
      reached or when scheduled_run_count >= max_run_count. If not specified,
      new runs will keep getting scheduled until this Schedule is paused or
      deleted. Already scheduled runs will be allowed to complete. Unset if
      not specified.
    lastPauseTime: Output only. Timestamp when this Schedule was last paused.
      Unset if never paused.
    lastResumeTime: Output only. Timestamp when this Schedule was last
      resumed. Unset if never resumed from pause.
    lastScheduledRunResponse: Output only. Response of the last scheduled run.
      This is the response for starting the scheduled requests and not the
      execution of the operations/jobs created by the requests (if
      applicable). Unset if no run has been scheduled yet.
    maxConcurrentRunCount: Required. Maximum number of runs that can be
      started concurrently for this Schedule. This is the limit for starting
      the scheduled requests and not the execution of the operations/jobs
      created by the requests (if applicable).
    maxRunCount: Optional. Maximum run count of the schedule. If specified,
      The schedule will be completed when either started_run_count >=
      max_run_count or when end_time is reached. If not specified, new runs
      will keep getting scheduled until this Schedule is paused or deleted.
      Already scheduled runs will be allowed to complete. Unset if not
      specified.
    name: Immutable. The resource name of the Schedule.
    nextRunTime: Output only. Timestamp when this Schedule should schedule the
      next run. Having a next_run_time in the past means the runs are being
      started behind schedule.
    startTime: Optional. Timestamp after which the first run can be scheduled.
      Default to Schedule create time if not specified.
    startedRunCount: Output only. The number of runs started by this schedule.
    state: Output only. The state of this Schedule.
    updateTime: Output only. Timestamp when this Schedule was updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of this Schedule.

    Values:
      STATE_UNSPECIFIED: Unspecified.
      ACTIVE: The Schedule is active. Runs are being scheduled on the user-
        specified timespec.
      PAUSED: The schedule is paused. No new runs will be created until the
        schedule is resumed. Already started runs will be allowed to complete.
      COMPLETED: The Schedule is completed. No new runs will be scheduled.
        Already started runs will be allowed to complete. Schedules in
        completed state cannot be paused or resumed.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PAUSED = 2
        COMPLETED = 3
    allowQueueing = _messages.BooleanField(1)
    catchUp = _messages.BooleanField(2)
    createPipelineJobRequest = _messages.MessageField('GoogleCloudAiplatformV1CreatePipelineJobRequest', 3)
    createTime = _messages.StringField(4)
    cron = _messages.StringField(5)
    displayName = _messages.StringField(6)
    endTime = _messages.StringField(7)
    lastPauseTime = _messages.StringField(8)
    lastResumeTime = _messages.StringField(9)
    lastScheduledRunResponse = _messages.MessageField('GoogleCloudAiplatformV1ScheduleRunResponse', 10)
    maxConcurrentRunCount = _messages.IntegerField(11)
    maxRunCount = _messages.IntegerField(12)
    name = _messages.StringField(13)
    nextRunTime = _messages.StringField(14)
    startTime = _messages.StringField(15)
    startedRunCount = _messages.IntegerField(16)
    state = _messages.EnumField('StateValueValuesEnum', 17)
    updateTime = _messages.StringField(18)