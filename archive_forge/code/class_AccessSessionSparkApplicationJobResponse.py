from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessSessionSparkApplicationJobResponse(_messages.Message):
    """Details of a particular job associated with Spark Application

  Fields:
    jobData: Output only. Data corresponding to a spark job.
  """
    jobData = _messages.MessageField('JobData', 1)