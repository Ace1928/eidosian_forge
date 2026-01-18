from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEnvironmentsUsersSessionsContextsPatchRequest(_messages.Message):
    """A DialogflowProjectsAgentEnvironmentsUsersSessionsContextsPatchRequest
  object.

  Fields:
    googleCloudDialogflowV2Context: A GoogleCloudDialogflowV2Context resource
      to be passed as the request body.
    name: Required. The unique identifier of the context. Format:
      `projects//agent/sessions//contexts/`, or
      `projects//agent/environments//users//sessions//contexts/`. The `Context
      ID` is always converted to lowercase, may only contain characters in
      `a-zA-Z0-9_-%` and may be at most 250 bytes long. If `Environment ID` is
      not specified, we assume default 'draft' environment. If `User ID` is
      not specified, we assume default '-' user. The following context names
      are reserved for internal use by Dialogflow. You should not use these
      contexts or create contexts with these names: * `__system_counters__` *
      `*_id_dialog_context` * `*_dialog_params_size`
    updateMask: Optional. The mask to control which fields get updated.
  """
    googleCloudDialogflowV2Context = _messages.MessageField('GoogleCloudDialogflowV2Context', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)