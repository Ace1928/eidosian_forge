from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalCatalogDatasetOptions(_messages.Message):
    """Options defining open source compatible datasets living in the BigQuery
  catalog. Contains metadata of open source database, schema or namespace
  represented by the current dataset.

  Messages:
    ParametersValue: Optional. A map of key value pairs defining the
      parameters and properties of the open source schema. Maximum size of
      2Mib.

  Fields:
    defaultStorageLocationUri: Optional. The storage location URI for all
      tables in the dataset. Equivalent to hive metastore's database
      locationUri. Maximum length of 1024 characters.
    parameters: Optional. A map of key value pairs defining the parameters and
      properties of the open source schema. Maximum size of 2Mib.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """Optional. A map of key value pairs defining the parameters and
    properties of the open source schema. Maximum size of 2Mib.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Additional properties of type ParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    defaultStorageLocationUri = _messages.StringField(1)
    parameters = _messages.MessageField('ParametersValue', 2)