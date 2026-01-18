from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeqMapTask(_messages.Message):
    """Describes a particular function to invoke.

  Messages:
    UserFnValue: The user function to invoke.

  Fields:
    inputs: Information about each of the inputs.
    name: The user-provided name of the SeqDo operation.
    outputInfos: Information about each of the outputs.
    stageName: System-defined name of the stage containing the SeqDo
      operation. Unique across the workflow.
    systemName: System-defined name of the SeqDo operation. Unique across the
      workflow.
    userFn: The user function to invoke.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserFnValue(_messages.Message):
        """The user function to invoke.

    Messages:
      AdditionalProperty: An additional property for a UserFnValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserFnValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    inputs = _messages.MessageField('SideInputInfo', 1, repeated=True)
    name = _messages.StringField(2)
    outputInfos = _messages.MessageField('SeqMapTaskOutputInfo', 3, repeated=True)
    stageName = _messages.StringField(4)
    systemName = _messages.StringField(5)
    userFn = _messages.MessageField('UserFnValue', 6)