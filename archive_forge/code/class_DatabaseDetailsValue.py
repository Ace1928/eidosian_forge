from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DatabaseDetailsValue(_messages.Message):
    """Optional. Backup details per database in Cloud Storage.

    Messages:
      AdditionalProperty: An additional property for a DatabaseDetailsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DatabaseDetailsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DatabaseDetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A SqlServerDatabaseDetails attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('SqlServerDatabaseDetails', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)