from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityCaaIntelFrontendAccessLevelExplanation(_messages.Message):
    """Explanation of access level, including the original access level defined
  by customers, evaluation results and metadata NextTAG: 12

  Enums:
    AccessLevelStateValueValuesEnum: Evaluation state of an access level

  Messages:
    NodeMapValue: Map between node.id and cel node Node id: Expr.id
      (google/api/expr/syntax.proto)
    NodeNegTroubleshootingMetadataMapValue: Map between node.id and
      troubleshooting metadata of this node when the state of this access
      level is expected to be not_granted
    NodePosTroubleshootingMetadataMapValue: Map between node.id and
      troubleshooting metadata of this node when the state of this access
      level is expected to be granted

  Fields:
    accessLevelState: Evaluation state of an access level
    basicLevelExplanation: A IdentityCaaIntelFrontendBasicLevelExplanation
      attribute.
    customLevelExplanation: A IdentityCaaIntelFrontendCustomLevelExplanation
      attribute.
    name: Resource name for the Access Level. Format:
      `accessPolicies/{policy_id}/accessLevels/{short_name}`
    nodeMap: Map between node.id and cel node Node id: Expr.id
      (google/api/expr/syntax.proto)
    nodeNegTroubleshootingMetadataMap: Map between node.id and troubleshooting
      metadata of this node when the state of this access level is expected to
      be not_granted
    nodePosTroubleshootingMetadataMap: Map between node.id and troubleshooting
      metadata of this node when the state of this access level is expected to
      be granted
    title: Access Level's title
  """

    class AccessLevelStateValueValuesEnum(_messages.Enum):
        """Evaluation state of an access level

    Values:
      ACCESS_LEVEL_STATE_UNSPECIFIED: Reserved
      ACCESS_LEVEL_STATE_GRANTED: The access level state is granted
      ACCESS_LEVEL_STATE_NOT_GRANTED: The access level state is not granted
      ACCESS_LEVEL_STATE_ERROR: Encounter error when evaluating this access
        level. Note that such error is on the critical path that blocks the
        evaluation; e.g. False || -> ACCESS_LEVEL_STATE_NOT_GRANTED True && ->
        ACCESS_LEVEL_STATE_ERROR
      ACCESS_LEVEL_NOT_EXIST: The access level doesn't exist
    """
        ACCESS_LEVEL_STATE_UNSPECIFIED = 0
        ACCESS_LEVEL_STATE_GRANTED = 1
        ACCESS_LEVEL_STATE_NOT_GRANTED = 2
        ACCESS_LEVEL_STATE_ERROR = 3
        ACCESS_LEVEL_NOT_EXIST = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NodeMapValue(_messages.Message):
        """Map between node.id and cel node Node id: Expr.id
    (google/api/expr/syntax.proto)

    Messages:
      AdditionalProperty: An additional property for a NodeMapValue object.

    Fields:
      additionalProperties: Additional properties of type NodeMapValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NodeMapValue object.

      Fields:
        key: Name of the additional property.
        value: A IdentityCaaIntelFrontendCelNode attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('IdentityCaaIntelFrontendCelNode', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NodeNegTroubleshootingMetadataMapValue(_messages.Message):
        """Map between node.id and troubleshooting metadata of this node when the
    state of this access level is expected to be not_granted

    Messages:
      AdditionalProperty: An additional property for a
        NodeNegTroubleshootingMetadataMapValue object.

    Fields:
      additionalProperties: Additional properties of type
        NodeNegTroubleshootingMetadataMapValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NodeNegTroubleshootingMetadataMapValue
      object.

      Fields:
        key: Name of the additional property.
        value: A IdentityCaaIntelFrontendTroubleshootingMetadata attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('IdentityCaaIntelFrontendTroubleshootingMetadata', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NodePosTroubleshootingMetadataMapValue(_messages.Message):
        """Map between node.id and troubleshooting metadata of this node when the
    state of this access level is expected to be granted

    Messages:
      AdditionalProperty: An additional property for a
        NodePosTroubleshootingMetadataMapValue object.

    Fields:
      additionalProperties: Additional properties of type
        NodePosTroubleshootingMetadataMapValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NodePosTroubleshootingMetadataMapValue
      object.

      Fields:
        key: Name of the additional property.
        value: A IdentityCaaIntelFrontendTroubleshootingMetadata attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('IdentityCaaIntelFrontendTroubleshootingMetadata', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    accessLevelState = _messages.EnumField('AccessLevelStateValueValuesEnum', 1)
    basicLevelExplanation = _messages.MessageField('IdentityCaaIntelFrontendBasicLevelExplanation', 2)
    customLevelExplanation = _messages.MessageField('IdentityCaaIntelFrontendCustomLevelExplanation', 3)
    name = _messages.StringField(4)
    nodeMap = _messages.MessageField('NodeMapValue', 5)
    nodeNegTroubleshootingMetadataMap = _messages.MessageField('NodeNegTroubleshootingMetadataMapValue', 6)
    nodePosTroubleshootingMetadataMap = _messages.MessageField('NodePosTroubleshootingMetadataMapValue', 7)
    title = _messages.StringField(8)