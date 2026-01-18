from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2Environment(_messages.Message):
    """You can create multiple versions of your agent and publish them to
  separate environments. When you edit an agent, you are editing the draft
  agent. At any point, you can save the draft agent as an agent version, which
  is an immutable snapshot of your agent. When you save the draft agent, it is
  published to the default environment. When you create agent versions, you
  can publish them to custom environments. You can create a variety of custom
  environments for: - testing - development - production - etc. For more
  information, see the [versions and environments
  guide](https://cloud.google.com/dialogflow/docs/agents-versions).

  Enums:
    StateValueValuesEnum: Output only. The state of this environment. This
      field is read-only, i.e., it cannot be set by create and update methods.

  Fields:
    agentVersion: Optional. The agent version loaded into this environment.
      Supported formats: - `projects//agent/versions/` -
      `projects//locations//agent/versions/`
    description: Optional. The developer-provided description for this
      environment. The maximum length is 500 characters. If exceeded, the
      request is rejected.
    fulfillment: Optional. The fulfillment settings to use for this
      environment.
    name: Output only. The unique identifier of this agent environment.
      Supported formats: - `projects//agent/environments/` -
      `projects//locations//agent/environments/` The environment ID for the
      default environment is `-`.
    state: Output only. The state of this environment. This field is read-
      only, i.e., it cannot be set by create and update methods.
    textToSpeechSettings: Optional. Text to speech settings for this
      environment.
    updateTime: Output only. The last update time of this environment. This
      field is read-only, i.e., it cannot be set by create and update methods.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of this environment. This field is read-only,
    i.e., it cannot be set by create and update methods.

    Values:
      STATE_UNSPECIFIED: Not specified. This value is not used.
      STOPPED: Stopped.
      LOADING: Loading.
      RUNNING: Running.
    """
        STATE_UNSPECIFIED = 0
        STOPPED = 1
        LOADING = 2
        RUNNING = 3
    agentVersion = _messages.StringField(1)
    description = _messages.StringField(2)
    fulfillment = _messages.MessageField('GoogleCloudDialogflowV2Fulfillment', 3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    textToSpeechSettings = _messages.MessageField('GoogleCloudDialogflowV2TextToSpeechSettings', 6)
    updateTime = _messages.StringField(7)