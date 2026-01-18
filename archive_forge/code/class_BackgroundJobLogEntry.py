from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackgroundJobLogEntry(_messages.Message):
    """Execution log of a background job.

  Enums:
    CompletionStateValueValuesEnum: Output only. Job completion state, i.e.
      the final state after the job completed.
    JobTypeValueValuesEnum: The type of job that was executed.

  Fields:
    applyJobDetails: Output only. Apply job details.
    completionComment: Output only. Job completion comment, such as how many
      entities were seeded, how many warnings were found during conversion,
      and similar information.
    completionState: Output only. Job completion state, i.e. the final state
      after the job completed.
    convertJobDetails: Output only. Convert job details.
    finishTime: The timestamp when the background job was finished.
    id: The background job log entry ID.
    importRulesJobDetails: Output only. Import rules job details.
    jobType: The type of job that was executed.
    requestAutocommit: Output only. Whether the client requested the
      conversion workspace to be committed after a successful completion of
      the job.
    seedJobDetails: Output only. Seed job details.
    startTime: The timestamp when the background job was started.
  """

    class CompletionStateValueValuesEnum(_messages.Enum):
        """Output only. Job completion state, i.e. the final state after the job
    completed.

    Values:
      JOB_COMPLETION_STATE_UNSPECIFIED: The status is not specified. This
        state is used when job is not yet finished.
      SUCCEEDED: Success.
      FAILED: Error.
    """
        JOB_COMPLETION_STATE_UNSPECIFIED = 0
        SUCCEEDED = 1
        FAILED = 2

    class JobTypeValueValuesEnum(_messages.Enum):
        """The type of job that was executed.

    Values:
      BACKGROUND_JOB_TYPE_UNSPECIFIED: Unspecified background job type.
      BACKGROUND_JOB_TYPE_SOURCE_SEED: Job to seed from the source database.
      BACKGROUND_JOB_TYPE_CONVERT: Job to convert the source database into a
        draft of the destination database.
      BACKGROUND_JOB_TYPE_APPLY_DESTINATION: Job to apply the draft tree onto
        the destination.
      BACKGROUND_JOB_TYPE_IMPORT_RULES_FILE: Job to import and convert mapping
        rules from an external source such as an ora2pg config file.
    """
        BACKGROUND_JOB_TYPE_UNSPECIFIED = 0
        BACKGROUND_JOB_TYPE_SOURCE_SEED = 1
        BACKGROUND_JOB_TYPE_CONVERT = 2
        BACKGROUND_JOB_TYPE_APPLY_DESTINATION = 3
        BACKGROUND_JOB_TYPE_IMPORT_RULES_FILE = 4
    applyJobDetails = _messages.MessageField('ApplyJobDetails', 1)
    completionComment = _messages.StringField(2)
    completionState = _messages.EnumField('CompletionStateValueValuesEnum', 3)
    convertJobDetails = _messages.MessageField('ConvertJobDetails', 4)
    finishTime = _messages.StringField(5)
    id = _messages.StringField(6)
    importRulesJobDetails = _messages.MessageField('ImportRulesJobDetails', 7)
    jobType = _messages.EnumField('JobTypeValueValuesEnum', 8)
    requestAutocommit = _messages.BooleanField(9)
    seedJobDetails = _messages.MessageField('SeedJobDetails', 10)
    startTime = _messages.StringField(11)