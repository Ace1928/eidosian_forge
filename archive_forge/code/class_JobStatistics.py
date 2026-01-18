from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobStatistics(_messages.Message):
    """A JobStatistics object.

  Fields:
    creationTime: [Output-only] Creation time of this job, in milliseconds
      since the epoch. This field will be present on all jobs.
    endTime: [Output-only] End time of this job, in milliseconds since the
      epoch. This field will be present whenever a job is in the DONE state.
    extract: [Output-only] Statistics for an extract job.
    load: [Output-only] Statistics for a load job.
    query: [Output-only] Statistics for a query job.
    startTime: [Output-only] Start time of this job, in milliseconds since the
      epoch. This field will be present when the job transitions from the
      PENDING state to either RUNNING or DONE.
    totalBytesProcessed: [Output-only] [Deprecated] Use the bytes processed in
      the query statistics instead.
  """
    creationTime = _messages.IntegerField(1)
    endTime = _messages.IntegerField(2)
    extract = _messages.MessageField('JobStatistics4', 3)
    load = _messages.MessageField('JobStatistics3', 4)
    query = _messages.MessageField('JobStatistics2', 5)
    startTime = _messages.IntegerField(6)
    totalBytesProcessed = _messages.IntegerField(7)