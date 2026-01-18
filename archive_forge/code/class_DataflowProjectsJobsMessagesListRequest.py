from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsJobsMessagesListRequest(_messages.Message):
    """A DataflowProjectsJobsMessagesListRequest object.

  Enums:
    MinimumImportanceValueValuesEnum: Filter to only get messages with
      importance >= level

  Fields:
    endTime: Return only messages with timestamps < end_time. The default is
      now (i.e. return up to the latest messages available).
    jobId: The job to get messages about.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the job specified by job_id.
    minimumImportance: Filter to only get messages with importance >= level
    pageSize: If specified, determines the maximum number of messages to
      return. If unspecified, the service may choose an appropriate default,
      or may return an arbitrarily large number of results.
    pageToken: If supplied, this should be the value of next_page_token
      returned by an earlier call. This will cause the next page of results to
      be returned.
    projectId: A project id.
    startTime: If specified, return only messages with timestamps >=
      start_time. The default is the job creation time (i.e. beginning of
      messages).
  """

    class MinimumImportanceValueValuesEnum(_messages.Enum):
        """Filter to only get messages with importance >= level

    Values:
      JOB_MESSAGE_IMPORTANCE_UNKNOWN: The message importance isn't specified,
        or is unknown.
      JOB_MESSAGE_DEBUG: The message is at the 'debug' level: typically only
        useful for software engineers working on the code the job is running.
        Typically, Dataflow pipeline runners do not display log messages at
        this level by default.
      JOB_MESSAGE_DETAILED: The message is at the 'detailed' level: somewhat
        verbose, but potentially useful to users. Typically, Dataflow pipeline
        runners do not display log messages at this level by default. These
        messages are displayed by default in the Dataflow monitoring UI.
      JOB_MESSAGE_BASIC: The message is at the 'basic' level: useful for
        keeping track of the execution of a Dataflow pipeline. Typically,
        Dataflow pipeline runners display log messages at this level by
        default, and these messages are displayed by default in the Dataflow
        monitoring UI.
      JOB_MESSAGE_WARNING: The message is at the 'warning' level: indicating a
        condition pertaining to a job which may require human intervention.
        Typically, Dataflow pipeline runners display log messages at this
        level by default, and these messages are displayed by default in the
        Dataflow monitoring UI.
      JOB_MESSAGE_ERROR: The message is at the 'error' level: indicating a
        condition preventing a job from succeeding. Typically, Dataflow
        pipeline runners display log messages at this level by default, and
        these messages are displayed by default in the Dataflow monitoring UI.
    """
        JOB_MESSAGE_IMPORTANCE_UNKNOWN = 0
        JOB_MESSAGE_DEBUG = 1
        JOB_MESSAGE_DETAILED = 2
        JOB_MESSAGE_BASIC = 3
        JOB_MESSAGE_WARNING = 4
        JOB_MESSAGE_ERROR = 5
    endTime = _messages.StringField(1)
    jobId = _messages.StringField(2, required=True)
    location = _messages.StringField(3)
    minimumImportance = _messages.EnumField('MinimumImportanceValueValuesEnum', 4)
    pageSize = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(6)
    projectId = _messages.StringField(7, required=True)
    startTime = _messages.StringField(8)