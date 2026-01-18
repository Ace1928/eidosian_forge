from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecuteMutationRequest(_messages.Message):
    """The ExecuteMutation request to Firebase Data Connect.

  Messages:
    VariablesValue: Optional. Values for GraphQL variables provided in this
      request.

  Fields:
    operationName: Required. The name of the GraphQL operation name. Required
      because all Connector operations must be named. See
      https://graphql.org/learn/queries/#operation-name.
    variables: Optional. Values for GraphQL variables provided in this
      request.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class VariablesValue(_messages.Message):
        """Optional. Values for GraphQL variables provided in this request.

    Messages:
      AdditionalProperty: An additional property for a VariablesValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a VariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    operationName = _messages.StringField(1)
    variables = _messages.MessageField('VariablesValue', 2)