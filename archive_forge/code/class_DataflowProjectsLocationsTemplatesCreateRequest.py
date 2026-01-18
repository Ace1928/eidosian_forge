from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsTemplatesCreateRequest(_messages.Message):
    """A DataflowProjectsLocationsTemplatesCreateRequest object.

  Fields:
    createJobFromTemplateRequest: A CreateJobFromTemplateRequest resource to
      be passed as the request body.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints) to
      which to direct the request.
    projectId: Required. The ID of the Cloud Platform project that the job
      belongs to.
  """
    createJobFromTemplateRequest = _messages.MessageField('CreateJobFromTemplateRequest', 1)
    location = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)