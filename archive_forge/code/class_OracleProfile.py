from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OracleProfile(_messages.Message):
    """Oracle database profile.

  Messages:
    ConnectionAttributesValue: Connection string attributes

  Fields:
    connectionAttributes: Connection string attributes
    databaseService: Required. Database for the Oracle connection.
    hostname: Required. Hostname for the Oracle connection.
    oracleSslConfig: Optional. SSL configuration for the Oracle connection.
    password: Required. Password for the Oracle connection.
    port: Port for the Oracle connection, default value is 1521.
    username: Required. Username for the Oracle connection.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConnectionAttributesValue(_messages.Message):
        """Connection string attributes

    Messages:
      AdditionalProperty: An additional property for a
        ConnectionAttributesValue object.

    Fields:
      additionalProperties: Additional properties of type
        ConnectionAttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConnectionAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    connectionAttributes = _messages.MessageField('ConnectionAttributesValue', 1)
    databaseService = _messages.StringField(2)
    hostname = _messages.StringField(3)
    oracleSslConfig = _messages.MessageField('OracleSslConfig', 4)
    password = _messages.StringField(5)
    port = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    username = _messages.StringField(7)