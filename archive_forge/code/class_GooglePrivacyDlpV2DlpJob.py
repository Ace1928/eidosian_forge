from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DlpJob(_messages.Message):
    """Combines all of the information about a DLP job.

  Enums:
    StateValueValuesEnum: State of a job.
    TypeValueValuesEnum: The type of job.

  Fields:
    actionDetails: Events that should occur after the job has completed.
    createTime: Time when the job was created.
    endTime: Time when the job finished.
    errors: A stream of errors encountered running the job.
    inspectDetails: Results from inspecting a data source.
    jobTriggerName: If created by a job trigger, the resource name of the
      trigger that instantiated the job.
    lastModified: Time when the job was last modified by the system.
    name: The server-assigned name.
    riskDetails: Results from analyzing risk of a data source.
    startTime: Time when the job started.
    state: State of a job.
    type: The type of job.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of a job.

    Values:
      JOB_STATE_UNSPECIFIED: Unused.
      PENDING: The job has not yet started.
      RUNNING: The job is currently running. Once a job has finished it will
        transition to FAILED or DONE.
      DONE: The job is no longer running.
      CANCELED: The job was canceled before it could be completed.
      FAILED: The job had an error and did not complete.
      ACTIVE: The job is currently accepting findings via hybridInspect. A
        hybrid job in ACTIVE state may continue to have findings added to it
        through the calling of hybridInspect. After the job has finished no
        more calls to hybridInspect may be made. ACTIVE jobs can transition to
        DONE.
    """
        JOB_STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        DONE = 3
        CANCELED = 4
        FAILED = 5
        ACTIVE = 6

    class TypeValueValuesEnum(_messages.Enum):
        """The type of job.

    Values:
      DLP_JOB_TYPE_UNSPECIFIED: Defaults to INSPECT_JOB.
      INSPECT_JOB: The job inspected Google Cloud for sensitive data.
      RISK_ANALYSIS_JOB: The job executed a Risk Analysis computation.
    """
        DLP_JOB_TYPE_UNSPECIFIED = 0
        INSPECT_JOB = 1
        RISK_ANALYSIS_JOB = 2
    actionDetails = _messages.MessageField('GooglePrivacyDlpV2ActionDetails', 1, repeated=True)
    createTime = _messages.StringField(2)
    endTime = _messages.StringField(3)
    errors = _messages.MessageField('GooglePrivacyDlpV2Error', 4, repeated=True)
    inspectDetails = _messages.MessageField('GooglePrivacyDlpV2InspectDataSourceDetails', 5)
    jobTriggerName = _messages.StringField(6)
    lastModified = _messages.StringField(7)
    name = _messages.StringField(8)
    riskDetails = _messages.MessageField('GooglePrivacyDlpV2AnalyzeDataSourceRiskDetails', 9)
    startTime = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    type = _messages.EnumField('TypeValueValuesEnum', 12)