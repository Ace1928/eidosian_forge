from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsTemplatesCreateRequest(_messages.Message):
    """A DataflowProjectsTemplatesCreateRequest object.

  Fields:
    createJobFromTemplateRequest: A CreateJobFromTemplateRequest resource to
      be passed as the request body.
    projectId: Required. The ID of the Cloud Platform project that the job
      belongs to.
  """
    createJobFromTemplateRequest = _messages.MessageField('CreateJobFromTemplateRequest', 1)
    projectId = _messages.StringField(2, required=True)