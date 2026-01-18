from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsEntityTypesPatchRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsEntityTypesPa
  tchRequest object.

  Fields:
    googleCloudDialogflowV2SessionEntityType: A
      GoogleCloudDialogflowV2SessionEntityType resource to be passed as the
      request body.
    name: Required. The unique identifier of this session entity type. Format:
      `projects//agent/sessions//entityTypes/`, or
      `projects//agent/environments//users//sessions//entityTypes/`. If
      `Environment ID` is not specified, we assume default 'draft'
      environment. If `User ID` is not specified, we assume default '-' user.
      `` must be the display name of an existing entity type in the same agent
      that will be overridden or supplemented.
    updateMask: Optional. The mask to control which fields get updated.
  """
    googleCloudDialogflowV2SessionEntityType = _messages.MessageField('GoogleCloudDialogflowV2SessionEntityType', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)