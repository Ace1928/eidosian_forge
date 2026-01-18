from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentEnvironmentsCreateRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentEnvironmentsCreateRequest object.

  Fields:
    environmentId: Required. The unique id of the new environment.
    googleCloudDialogflowV2Environment: A GoogleCloudDialogflowV2Environment
      resource to be passed as the request body.
    parent: Required. The agent to create an environment for. Supported
      formats: - `projects//agent` - `projects//locations//agent`
  """
    environmentId = _messages.StringField(1)
    googleCloudDialogflowV2Environment = _messages.MessageField('GoogleCloudDialogflowV2Environment', 2)
    parent = _messages.StringField(3, required=True)