from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryJobsInsertRequest(_messages.Message):
    """A BigqueryJobsInsertRequest object.

  Fields:
    job: A Job resource to be passed as the request body.
    projectId: Project ID of the project that will be billed for the job
  """
    job = _messages.MessageField('Job', 1)
    projectId = _messages.StringField(2, required=True)