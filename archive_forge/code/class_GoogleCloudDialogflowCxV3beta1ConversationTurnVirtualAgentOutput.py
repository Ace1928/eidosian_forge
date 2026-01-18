from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ConversationTurnVirtualAgentOutput(_messages.Message):
    """The output from the virtual agent.

  Messages:
    DiagnosticInfoValue: Required. Input only. The diagnostic info output for
      the turn. Required to calculate the testing coverage.
    SessionParametersValue: The session parameters available to the bot at
      this point.

  Fields:
    currentPage: The Page on which the utterance was spoken. Only name and
      displayName will be set.
    diagnosticInfo: Required. Input only. The diagnostic info output for the
      turn. Required to calculate the testing coverage.
    differences: Output only. If this is part of a result conversation turn,
      the list of differences between the original run and the replay for this
      output, if any.
    sessionParameters: The session parameters available to the bot at this
      point.
    status: Response error from the agent in the test result. If set, other
      output is empty.
    textResponses: The text responses from the agent for the turn.
    triggeredIntent: The Intent that triggered the response. Only name and
      displayName will be set.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DiagnosticInfoValue(_messages.Message):
        """Required. Input only. The diagnostic info output for the turn.
    Required to calculate the testing coverage.

    Messages:
      AdditionalProperty: An additional property for a DiagnosticInfoValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DiagnosticInfoValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SessionParametersValue(_messages.Message):
        """The session parameters available to the bot at this point.

    Messages:
      AdditionalProperty: An additional property for a SessionParametersValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SessionParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    currentPage = _messages.MessageField('GoogleCloudDialogflowCxV3beta1Page', 1)
    diagnosticInfo = _messages.MessageField('DiagnosticInfoValue', 2)
    differences = _messages.MessageField('GoogleCloudDialogflowCxV3beta1TestRunDifference', 3, repeated=True)
    sessionParameters = _messages.MessageField('SessionParametersValue', 4)
    status = _messages.MessageField('GoogleRpcStatus', 5)
    textResponses = _messages.MessageField('GoogleCloudDialogflowCxV3beta1ResponseMessageText', 6, repeated=True)
    triggeredIntent = _messages.MessageField('GoogleCloudDialogflowCxV3beta1Intent', 7)