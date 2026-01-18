from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GraphqlRequest(_messages.Message):
    """The GraphQL request to Firebase Data Connect. It strives to match the
  GraphQL over HTTP spec. https://github.com/graphql/graphql-over-
  http/blob/main/spec/GraphQLOverHTTP.md#post

  Messages:
    VariablesValue: Optional. Values for GraphQL variables provided in this
      request.

  Fields:
    extensions: Optional. Additional GraphQL request information.
    operationName: Optional. The name of the GraphQL operation name. Required
      only if `query` contains multiple operations. See
      https://graphql.org/learn/queries/#operation-name.
    query: Required. The GraphQL query document source.
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
    extensions = _messages.MessageField('GraphqlRequestExtensions', 1)
    operationName = _messages.StringField(2)
    query = _messages.StringField(3)
    variables = _messages.MessageField('VariablesValue', 4)