from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class TableDefinitionsValue(_messages.Message):
    """[Optional] If querying an external data source outside of BigQuery,
    describes the data format, location and other properties of the data
    source. By defining these properties, the data source can then be queried
    as if it were a standard BigQuery table.

    Messages:
      AdditionalProperty: An additional property for a TableDefinitionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        TableDefinitionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a TableDefinitionsValue object.

      Fields:
        key: Name of the additional property.
        value: A ExternalDataConfiguration attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ExternalDataConfiguration', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)