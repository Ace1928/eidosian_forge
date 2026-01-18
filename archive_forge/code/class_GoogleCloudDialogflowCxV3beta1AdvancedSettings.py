from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1AdvancedSettings(_messages.Message):
    """Hierarchical advanced settings for
  agent/flow/page/fulfillment/parameter. Settings exposed at lower level
  overrides the settings exposed at higher level. Overriding occurs at the
  sub-setting level. For example, the playback_interruption_settings at
  fulfillment level only overrides the playback_interruption_settings at the
  agent level, leaving other settings at the agent level unchanged. DTMF
  settings does not override each other. DTMF settings set at different levels
  define DTMF detections running in parallel. Hierarchy:
  Agent->Flow->Page->Fulfillment/Parameter.

  Fields:
    audioExportGcsDestination: If present, incoming audio is exported by
      Dialogflow to the configured Google Cloud Storage destination. Exposed
      at the following levels: - Agent level - Flow level
    dtmfSettings: Settings for DTMF. Exposed at the following levels: - Agent
      level - Flow level - Page level - Parameter level.
    loggingSettings: Settings for logging. Settings for Dialogflow History,
      Contact Center messages, StackDriver logs, and speech logging. Exposed
      at the following levels: - Agent level.
    speechSettings: Settings for speech to text detection. Exposed at the
      following levels: - Agent level - Flow level - Page level - Parameter
      level
  """
    audioExportGcsDestination = _messages.MessageField('GoogleCloudDialogflowCxV3beta1GcsDestination', 1)
    dtmfSettings = _messages.MessageField('GoogleCloudDialogflowCxV3beta1AdvancedSettingsDtmfSettings', 2)
    loggingSettings = _messages.MessageField('GoogleCloudDialogflowCxV3beta1AdvancedSettingsLoggingSettings', 3)
    speechSettings = _messages.MessageField('GoogleCloudDialogflowCxV3beta1AdvancedSettingsSpeechSettings', 4)