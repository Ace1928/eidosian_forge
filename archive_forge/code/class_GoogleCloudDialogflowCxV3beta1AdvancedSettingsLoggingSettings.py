from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1AdvancedSettingsLoggingSettings(_messages.Message):
    """Define behaviors on logging.

  Fields:
    enableInteractionLogging: If true, DF Interaction logging is currently
      enabled.
    enableStackdriverLogging: If true, StackDriver logging is currently
      enabled.
  """
    enableInteractionLogging = _messages.BooleanField(1)
    enableStackdriverLogging = _messages.BooleanField(2)