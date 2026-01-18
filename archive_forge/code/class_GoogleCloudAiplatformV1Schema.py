from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Schema(_messages.Message):
    """Schema is used to define the format of input/output data. Represents a
  select subset of an [OpenAPI 3.0 schema
  object](https://spec.openapis.org/oas/v3.0.3#schema). More fields may be
  added in the future as needed.

  Enums:
    TypeValueValuesEnum: Optional. The type of the data.

  Messages:
    PropertiesValue: Optional. SCHEMA FIELDS FOR TYPE OBJECT Properties of
      Type.OBJECT.

  Fields:
    default: Optional. Default value of the data.
    description: Optional. The description of the data.
    enum: Optional. Possible values of the element of Type.STRING with enum
      format. For example we can define an Enum Direction as : {type:STRING,
      format:enum, enum:["EAST", NORTH", "SOUTH", "WEST"]}
    example: Optional. Example of the object. Will only populated when the
      object is the root.
    format: Optional. The format of the data. Supported formats: for NUMBER
      type: "float", "double" for INTEGER type: "int32", "int64" for STRING
      type: "email", "byte", etc
    items: Optional. SCHEMA FIELDS FOR TYPE ARRAY Schema of the elements of
      Type.ARRAY.
    maxItems: Optional. Maximum number of the elements for Type.ARRAY.
    maxLength: Optional. Maximum length of the Type.STRING
    maxProperties: Optional. Maximum number of the properties for Type.OBJECT.
    maximum: Optional. Maximum value of the Type.INTEGER and Type.NUMBER
    minItems: Optional. Minimum number of the elements for Type.ARRAY.
    minLength: Optional. SCHEMA FIELDS FOR TYPE STRING Minimum length of the
      Type.STRING
    minProperties: Optional. Minimum number of the properties for Type.OBJECT.
    minimum: Optional. SCHEMA FIELDS FOR TYPE INTEGER and NUMBER Minimum value
      of the Type.INTEGER and Type.NUMBER
    nullable: Optional. Indicates if the value may be null.
    pattern: Optional. Pattern of the Type.STRING to restrict a string to a
      regular expression.
    properties: Optional. SCHEMA FIELDS FOR TYPE OBJECT Properties of
      Type.OBJECT.
    required: Optional. Required properties of Type.OBJECT.
    title: Optional. The title of the Schema.
    type: Optional. The type of the data.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Optional. The type of the data.

    Values:
      TYPE_UNSPECIFIED: Not specified, should not be used.
      STRING: OpenAPI string type
      NUMBER: OpenAPI number type
      INTEGER: OpenAPI integer type
      BOOLEAN: OpenAPI boolean type
      ARRAY: OpenAPI array type
      OBJECT: OpenAPI object type
    """
        TYPE_UNSPECIFIED = 0
        STRING = 1
        NUMBER = 2
        INTEGER = 3
        BOOLEAN = 4
        ARRAY = 5
        OBJECT = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Optional. SCHEMA FIELDS FOR TYPE OBJECT Properties of Type.OBJECT.

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1Schema attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1Schema', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    default = _messages.MessageField('extra_types.JsonValue', 1)
    description = _messages.StringField(2)
    enum = _messages.StringField(3, repeated=True)
    example = _messages.MessageField('extra_types.JsonValue', 4)
    format = _messages.StringField(5)
    items = _messages.MessageField('GoogleCloudAiplatformV1Schema', 6)
    maxItems = _messages.IntegerField(7)
    maxLength = _messages.IntegerField(8)
    maxProperties = _messages.IntegerField(9)
    maximum = _messages.FloatField(10)
    minItems = _messages.IntegerField(11)
    minLength = _messages.IntegerField(12)
    minProperties = _messages.IntegerField(13)
    minimum = _messages.FloatField(14)
    nullable = _messages.BooleanField(15)
    pattern = _messages.StringField(16)
    properties = _messages.MessageField('PropertiesValue', 17)
    required = _messages.StringField(18, repeated=True)
    title = _messages.StringField(19)
    type = _messages.EnumField('TypeValueValuesEnum', 20)