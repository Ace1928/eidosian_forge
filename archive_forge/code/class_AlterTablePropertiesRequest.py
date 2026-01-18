from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlterTablePropertiesRequest(_messages.Message):
    """Request message for DataprocMetastore.AlterTableProperties.

  Messages:
    PropertiesValue: A map that describes the desired values to mutate. If
      update_mask is empty, the properties will not update. Otherwise, the
      properties only alters the value whose associated paths exist in the
      update mask

  Fields:
    properties: A map that describes the desired values to mutate. If
      update_mask is empty, the properties will not update. Otherwise, the
      properties only alters the value whose associated paths exist in the
      update mask
    tableName: Required. The name of the table containing the properties
      you're altering in the following
      format.databases/{database_id}/tables/{table_id}
    updateMask: A field mask that specifies the metadata table properties that
      are overwritten by the update. Fields specified in the update_mask are
      relative to the resource (not to the full request). A field is
      overwritten if it is in the mask.For example, given the target
      properties: properties { a: 1 b: 2 } And an update properties:
      properties { a: 2 b: 3 c: 4 } then if the field mask is:paths:
      "properties.b", "properties.c"then the result will be: properties { a: 1
      b: 3 c: 4 }
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """A map that describes the desired values to mutate. If update_mask is
    empty, the properties will not update. Otherwise, the properties only
    alters the value whose associated paths exist in the update mask

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    properties = _messages.MessageField('PropertiesValue', 1)
    tableName = _messages.StringField(2)
    updateMask = _messages.StringField(3)