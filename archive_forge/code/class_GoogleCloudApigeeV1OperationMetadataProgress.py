from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1OperationMetadataProgress(_messages.Message):
    """Information about operation progress.

  Enums:
    StateValueValuesEnum: State of the operation.

  Messages:
    DetailsValue: The additional details of the progress.

  Fields:
    description: Description of the operation's progress.
    details: The additional details of the progress.
    percentDone: The percentage of the operation progress.
    state: State of the operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of the operation.

    Values:
      STATE_UNSPECIFIED: <no description>
      NOT_STARTED: <no description>
      IN_PROGRESS: <no description>
      FINISHED: <no description>
    """
        STATE_UNSPECIFIED = 0
        NOT_STARTED = 1
        IN_PROGRESS = 2
        FINISHED = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DetailsValue(_messages.Message):
        """The additional details of the progress.

    Messages:
      AdditionalProperty: An additional property for a DetailsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    description = _messages.StringField(1)
    details = _messages.MessageField('DetailsValue', 2)
    percentDone = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    state = _messages.EnumField('StateValueValuesEnum', 4)