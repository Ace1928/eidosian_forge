from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentVersionsCreateRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentVersionsCreateRequest object.

  Fields:
    googleCloudDialogflowV2Version: A GoogleCloudDialogflowV2Version resource
      to be passed as the request body.
    parent: Required. The agent to create a version for. Supported formats: -
      `projects//agent` - `projects//locations//agent`
  """
    googleCloudDialogflowV2Version = _messages.MessageField('GoogleCloudDialogflowV2Version', 1)
    parent = _messages.StringField(2, required=True)