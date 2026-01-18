from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentSessionsEntityTypesCreateRequest(_messages.Message):
    """A DialogflowProjectsAgentSessionsEntityTypesCreateRequest object.

  Fields:
    googleCloudDialogflowV2SessionEntityType: A
      GoogleCloudDialogflowV2SessionEntityType resource to be passed as the
      request body.
    parent: Required. The session to create a session entity type for. Format:
      `projects//agent/sessions/` or `projects//agent/environments//users//
      sessions/`. If `Environment ID` is not specified, we assume default
      'draft' environment. If `User ID` is not specified, we assume default
      '-' user.
  """
    googleCloudDialogflowV2SessionEntityType = _messages.MessageField('GoogleCloudDialogflowV2SessionEntityType', 1)
    parent = _messages.StringField(2, required=True)