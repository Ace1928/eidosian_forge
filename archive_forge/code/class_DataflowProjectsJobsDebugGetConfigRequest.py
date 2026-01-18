from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsJobsDebugGetConfigRequest(_messages.Message):
    """A DataflowProjectsJobsDebugGetConfigRequest object.

  Fields:
    getDebugConfigRequest: A GetDebugConfigRequest resource to be passed as
      the request body.
    jobId: The job id.
    projectId: The project id.
  """
    getDebugConfigRequest = _messages.MessageField('GetDebugConfigRequest', 1)
    jobId = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)