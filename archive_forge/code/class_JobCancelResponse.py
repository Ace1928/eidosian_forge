from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobCancelResponse(_messages.Message):
    """A JobCancelResponse object.

  Fields:
    job: The final state of the job.
    kind: The resource type of the response.
  """
    job = _messages.MessageField('Job', 1)
    kind = _messages.StringField(2, default=u'bigquery#jobCancelResponse')