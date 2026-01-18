from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeGmailContextualTrigger(_messages.Message):
    """Defines a trigger that fires when the open email meets a specific
  criteria. When the trigger fires, it executes a specific endpoint, usually
  in order to create new cards and update the UI.

  Fields:
    onTriggerFunction: Required. The name of the endpoint to call when a
      message matches the trigger.
    unconditional: Unconditional triggers are executed when any mail message
      is opened.
  """
    onTriggerFunction = _messages.StringField(1)
    unconditional = _messages.MessageField('GoogleAppsScriptTypeGmailUnconditionalTrigger', 2)