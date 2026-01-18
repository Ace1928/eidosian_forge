from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimesAssignRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimesAssignRequest object.

  Fields:
    googleCloudAiplatformV1AssignNotebookRuntimeRequest: A
      GoogleCloudAiplatformV1AssignNotebookRuntimeRequest resource to be
      passed as the request body.
    parent: Required. The resource name of the Location to get the
      NotebookRuntime assignment. Format:
      `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1AssignNotebookRuntimeRequest = _messages.MessageField('GoogleCloudAiplatformV1AssignNotebookRuntimeRequest', 1)
    parent = _messages.StringField(2, required=True)