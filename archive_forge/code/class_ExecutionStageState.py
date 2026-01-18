from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionStageState(_messages.Message):
    """A message describing the state of a particular execution stage.

  Enums:
    ExecutionStageStateValueValuesEnum: Executions stage states allow the same
      set of values as JobState.

  Fields:
    currentStateTime: The time at which the stage transitioned to this state.
    executionStageName: The name of the execution stage.
    executionStageState: Executions stage states allow the same set of values
      as JobState.
  """

    class ExecutionStageStateValueValuesEnum(_messages.Enum):
        """Executions stage states allow the same set of values as JobState.

    Values:
      JOB_STATE_UNKNOWN: The job's run state isn't specified.
      JOB_STATE_STOPPED: `JOB_STATE_STOPPED` indicates that the job has not
        yet started to run.
      JOB_STATE_RUNNING: `JOB_STATE_RUNNING` indicates that the job is
        currently running.
      JOB_STATE_DONE: `JOB_STATE_DONE` indicates that the job has successfully
        completed. This is a terminal job state. This state may be set by the
        Cloud Dataflow service, as a transition from `JOB_STATE_RUNNING`. It
        may also be set via a Cloud Dataflow `UpdateJob` call, if the job has
        not yet reached a terminal state.
      JOB_STATE_FAILED: `JOB_STATE_FAILED` indicates that the job has failed.
        This is a terminal job state. This state may only be set by the Cloud
        Dataflow service, and only as a transition from `JOB_STATE_RUNNING`.
      JOB_STATE_CANCELLED: `JOB_STATE_CANCELLED` indicates that the job has
        been explicitly cancelled. This is a terminal job state. This state
        may only be set via a Cloud Dataflow `UpdateJob` call, and only if the
        job has not yet reached another terminal state.
      JOB_STATE_UPDATED: `JOB_STATE_UPDATED` indicates that the job was
        successfully updated, meaning that this job was stopped and another
        job was started, inheriting state from this one. This is a terminal
        job state. This state may only be set by the Cloud Dataflow service,
        and only as a transition from `JOB_STATE_RUNNING`.
      JOB_STATE_DRAINING: `JOB_STATE_DRAINING` indicates that the job is in
        the process of draining. A draining job has stopped pulling from its
        input sources and is processing any data that remains in-flight. This
        state may be set via a Cloud Dataflow `UpdateJob` call, but only as a
        transition from `JOB_STATE_RUNNING`. Jobs that are draining may only
        transition to `JOB_STATE_DRAINED`, `JOB_STATE_CANCELLED`, or
        `JOB_STATE_FAILED`.
      JOB_STATE_DRAINED: `JOB_STATE_DRAINED` indicates that the job has been
        drained. A drained job terminated by stopping pulling from its input
        sources and processing any data that remained in-flight when draining
        was requested. This state is a terminal state, may only be set by the
        Cloud Dataflow service, and only as a transition from
        `JOB_STATE_DRAINING`.
      JOB_STATE_PENDING: `JOB_STATE_PENDING` indicates that the job has been
        created but is not yet running. Jobs that are pending may only
        transition to `JOB_STATE_RUNNING`, or `JOB_STATE_FAILED`.
      JOB_STATE_CANCELLING: `JOB_STATE_CANCELLING` indicates that the job has
        been explicitly cancelled and is in the process of stopping. Jobs that
        are cancelling may only transition to `JOB_STATE_CANCELLED` or
        `JOB_STATE_FAILED`.
      JOB_STATE_QUEUED: `JOB_STATE_QUEUED` indicates that the job has been
        created but is being delayed until launch. Jobs that are queued may
        only transition to `JOB_STATE_PENDING` or `JOB_STATE_CANCELLED`.
      JOB_STATE_RESOURCE_CLEANING_UP: `JOB_STATE_RESOURCE_CLEANING_UP`
        indicates that the batch job's associated resources are currently
        being cleaned up after a successful run. Currently, this is an opt-in
        feature, please reach out to Cloud support team if you are interested.
    """
        JOB_STATE_UNKNOWN = 0
        JOB_STATE_STOPPED = 1
        JOB_STATE_RUNNING = 2
        JOB_STATE_DONE = 3
        JOB_STATE_FAILED = 4
        JOB_STATE_CANCELLED = 5
        JOB_STATE_UPDATED = 6
        JOB_STATE_DRAINING = 7
        JOB_STATE_DRAINED = 8
        JOB_STATE_PENDING = 9
        JOB_STATE_CANCELLING = 10
        JOB_STATE_QUEUED = 11
        JOB_STATE_RESOURCE_CLEANING_UP = 12
    currentStateTime = _messages.StringField(1)
    executionStageName = _messages.StringField(2)
    executionStageState = _messages.EnumField('ExecutionStageStateValueValuesEnum', 3)