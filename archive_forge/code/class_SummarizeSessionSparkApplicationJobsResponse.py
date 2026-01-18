from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SummarizeSessionSparkApplicationJobsResponse(_messages.Message):
    """Summary of a Spark Application jobs.

  Fields:
    jobsSummary: Summary of a Spark Application Jobs
  """
    jobsSummary = _messages.MessageField('JobsSummary', 1)