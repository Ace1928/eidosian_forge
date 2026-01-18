from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AspectTypeMetadataTemplate(_messages.Message):
    """MetadataTemplate definition for AspectType

  Fields:
    annotations: Optional. Specifies annotations on this field.
    arrayItems: Optional. array_items needs to be set if the type is array.
      array_items can refer to a primitive field or a complex (record only)
      field. To specify a primitive field, just name and type needs to be set
      in the nested MetadataTemplate. The recommended value for the name field
      is item, as this is not used in the actual payload.
    constraints: Optional. Specifies the constraints on this field.
    enumValues: Optional. The list of values for an enum type. Needs to be
      defined if the type is enum.
    index: Optional. Index is used to encode Template messages. The value of
      index can range between 1 and 2,147,483,647. Index must be unique within
      all fields in a Template. (Nested Templates can reuse indexes). Once a
      Template is defined, the index cannot be changed, because it identifies
      the field in the actual storage format. Index is a mandatory field, but
      it is optional for top level fields, and map/array "values" definitions.
    mapItems: Optional. map_items needs to be set if the type is map.
      map_items can refer to a primitive field or a complex (record only)
      field. To specify a primitive field, just name and type needs to be set
      in the nested MetadataTemplate. The recommended value for the name field
      is item, as this is not used in the actual payload.
    name: Required. The name of the field.
    recordFields: Optional. Field definition, needs to be specified if the
      type is record. Defines the nested fields.
    type: Required. The datatype of this field. The following values are
      supported: Primitive types (string, integer, boolean, double, datetime);
      datetime must be of the format RFC3339 UTC "Zulu" (Examples:
      "2014-10-02T15:01:23Z" and "2014-10-02T15:01:23.045123456Z"). Complex
      types (enum, array, map, record).
    typeId: Optional. Id can be used if this definition of the field needs to
      be reused later. Id needs to be unique across the entire template. Id
      can only be specified if the field type is record.
    typeRef: Optional. A reference to another field definition (instead of an
      inline definition). The value must be equal to the value of an id field
      defined elsewhere in the MetadataTemplate. Only fields with type as
      record can refer to other fields.
  """
    annotations = _messages.MessageField('GoogleCloudDataplexV1AspectTypeMetadataTemplateAnnotations', 1)
    arrayItems = _messages.MessageField('GoogleCloudDataplexV1AspectTypeMetadataTemplate', 2)
    constraints = _messages.MessageField('GoogleCloudDataplexV1AspectTypeMetadataTemplateConstraints', 3)
    enumValues = _messages.MessageField('GoogleCloudDataplexV1AspectTypeMetadataTemplateEnumValue', 4, repeated=True)
    index = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    mapItems = _messages.MessageField('GoogleCloudDataplexV1AspectTypeMetadataTemplate', 6)
    name = _messages.StringField(7)
    recordFields = _messages.MessageField('GoogleCloudDataplexV1AspectTypeMetadataTemplate', 8, repeated=True)
    type = _messages.StringField(9)
    typeId = _messages.StringField(10)
    typeRef = _messages.StringField(11)