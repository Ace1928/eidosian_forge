from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1NasJobOutputMultiTrialJobOutput(_messages.Message):
    """The output of Multi-trial Neural Architecture Search (NAS) jobs.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of the trial.

  Fields:
    allMetrics: All objective metrics for this Neural Architecture Search
      (NAS) job.
    endTime: Output only. End time for the trial.
    finalMetric: The final objective metric seen for this Neural Architecture
      Search (NAS) job.
    nasParamsStr: The parameters that are associated with this Neural
      Architecture Search (NAS) job.
    startTime: Output only. Start time for the trial.
    state: Output only. The detailed state of the trial.
    trialId: The trial id for these results.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of the trial.

    Values:
      STATE_UNSPECIFIED: The job state is unspecified.
      QUEUED: The job has been just created and processing has not yet begun.
      PREPARING: The service is preparing to run the job.
      RUNNING: The job is in progress.
      SUCCEEDED: The job completed successfully.
      FAILED: The job failed. `error_message` should contain the details of
        the failure.
      CANCELLING: The job is being cancelled. `error_message` should describe
        the reason for the cancellation.
      CANCELLED: The job has been cancelled. `error_message` should describe
        the reason for the cancellation.
    """
        STATE_UNSPECIFIED = 0
        QUEUED = 1
        PREPARING = 2
        RUNNING = 3
        SUCCEEDED = 4
        FAILED = 5
        CANCELLING = 6
        CANCELLED = 7
    allMetrics = _messages.MessageField('GoogleCloudMlV1NasJobOutputMultiTrialJobOutputNasParameterMetric', 1, repeated=True)
    endTime = _messages.StringField(2)
    finalMetric = _messages.MessageField('GoogleCloudMlV1NasJobOutputMultiTrialJobOutputNasParameterMetric', 3)
    nasParamsStr = _messages.StringField(4)
    startTime = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    trialId = _messages.StringField(7)