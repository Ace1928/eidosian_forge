from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudFunctionsV2betaStage(_messages.Message):
    """Each Stage of the deployment process

  Enums:
    NameValueValuesEnum: Name of the Stage. This will be unique for each
      Stage.
    StateValueValuesEnum: Current state of the Stage

  Fields:
    message: Message describing the Stage
    name: Name of the Stage. This will be unique for each Stage.
    resource: Resource of the Stage
    resourceUri: Link to the current Stage resource
    state: Current state of the Stage
    stateMessages: State messages from the current Stage.
  """

    class NameValueValuesEnum(_messages.Enum):
        """Name of the Stage. This will be unique for each Stage.

    Values:
      NAME_UNSPECIFIED: Not specified. Invalid name.
      ARTIFACT_REGISTRY: Artifact Regsitry Stage
      BUILD: Build Stage
      SERVICE: Service Stage
      TRIGGER: Trigger Stage
      SERVICE_ROLLBACK: Service Rollback Stage
      TRIGGER_ROLLBACK: Trigger Rollback Stage
    """
        NAME_UNSPECIFIED = 0
        ARTIFACT_REGISTRY = 1
        BUILD = 2
        SERVICE = 3
        TRIGGER = 4
        SERVICE_ROLLBACK = 5
        TRIGGER_ROLLBACK = 6

    class StateValueValuesEnum(_messages.Enum):
        """Current state of the Stage

    Values:
      STATE_UNSPECIFIED: Not specified. Invalid state.
      NOT_STARTED: Stage has not started.
      IN_PROGRESS: Stage is in progress.
      COMPLETE: Stage has completed.
    """
        STATE_UNSPECIFIED = 0
        NOT_STARTED = 1
        IN_PROGRESS = 2
        COMPLETE = 3
    message = _messages.StringField(1)
    name = _messages.EnumField('NameValueValuesEnum', 2)
    resource = _messages.StringField(3)
    resourceUri = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    stateMessages = _messages.MessageField('GoogleCloudFunctionsV2betaStateMessage', 6, repeated=True)