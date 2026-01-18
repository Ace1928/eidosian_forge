from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecuteQueryResponse(_messages.Message):
    """The ExecuteQuery response from Firebase Data Connect.

  Messages:
    DataValue: The result of executing the requested operation.

  Fields:
    data: The result of executing the requested operation.
    errors: Errors of this response.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DataValue(_messages.Message):
        """The result of executing the requested operation.

    Messages:
      AdditionalProperty: An additional property for a DataValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    data = _messages.MessageField('DataValue', 1)
    errors = _messages.MessageField('GraphqlError', 2, repeated=True)