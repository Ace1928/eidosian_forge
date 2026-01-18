from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEnvironmentsUsersSessionsContextsGetRequest(_messages.Message):
    """A DialogflowProjectsAgentEnvironmentsUsersSessionsContextsGetRequest
  object.

  Fields:
    name: Required. The name of the context. Format:
      `projects//agent/sessions//contexts/` or
      `projects//agent/environments//users//sessions//contexts/`. If
      `Environment ID` is not specified, we assume default 'draft'
      environment. If `User ID` is not specified, we assume default '-' user.
  """
    name = _messages.StringField(1, required=True)