from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesTasksCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesTasksCreateRequest object.

  Fields:
    googleCloudDataplexV1Task: A GoogleCloudDataplexV1Task resource to be
      passed as the request body.
    parent: Required. The resource name of the parent lake:
      projects/{project_number}/locations/{location_id}/lakes/{lake_id}.
    taskId: Required. Task identifier.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1Task = _messages.MessageField('GoogleCloudDataplexV1Task', 1)
    parent = _messages.StringField(2, required=True)
    taskId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)