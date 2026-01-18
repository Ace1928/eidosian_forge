from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Tensor(_messages.Message):
    """A tensor value type.

  Enums:
    DtypeValueValuesEnum: The data type of tensor.

  Messages:
    StructValValue: A map of string to tensor.

  Fields:
    boolVal: Type specific representations that make it easy to create tensor
      protos in all languages. Only the representation corresponding to
      "dtype" can be set. The values hold the flattened representation of the
      tensor in row major order. BOOL
    bytesVal: STRING
    doubleVal: DOUBLE
    dtype: The data type of tensor.
    floatVal: FLOAT
    int64Val: INT64
    intVal: INT_8 INT_16 INT_32
    listVal: A list of tensor values.
    shape: Shape of the tensor.
    stringVal: STRING
    structVal: A map of string to tensor.
    tensorVal: Serialized raw tensor content.
    uint64Val: UINT64
    uintVal: UINT8 UINT16 UINT32
  """

    class DtypeValueValuesEnum(_messages.Enum):
        """The data type of tensor.

    Values:
      DATA_TYPE_UNSPECIFIED: Not a legal value for DataType. Used to indicate
        a DataType field has not been set.
      BOOL: Data types that all computation devices are expected to be capable
        to support.
      STRING: <no description>
      FLOAT: <no description>
      DOUBLE: <no description>
      INT8: <no description>
      INT16: <no description>
      INT32: <no description>
      INT64: <no description>
      UINT8: <no description>
      UINT16: <no description>
      UINT32: <no description>
      UINT64: <no description>
    """
        DATA_TYPE_UNSPECIFIED = 0
        BOOL = 1
        STRING = 2
        FLOAT = 3
        DOUBLE = 4
        INT8 = 5
        INT16 = 6
        INT32 = 7
        INT64 = 8
        UINT8 = 9
        UINT16 = 10
        UINT32 = 11
        UINT64 = 12

    @encoding.MapUnrecognizedFields('additionalProperties')
    class StructValValue(_messages.Message):
        """A map of string to tensor.

    Messages:
      AdditionalProperty: An additional property for a StructValValue object.

    Fields:
      additionalProperties: Additional properties of type StructValValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a StructValValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1Tensor attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1Tensor', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    boolVal = _messages.BooleanField(1, repeated=True)
    bytesVal = _messages.BytesField(2, repeated=True)
    doubleVal = _messages.FloatField(3, repeated=True)
    dtype = _messages.EnumField('DtypeValueValuesEnum', 4)
    floatVal = _messages.FloatField(5, repeated=True, variant=_messages.Variant.FLOAT)
    int64Val = _messages.IntegerField(6, repeated=True)
    intVal = _messages.IntegerField(7, repeated=True, variant=_messages.Variant.INT32)
    listVal = _messages.MessageField('GoogleCloudAiplatformV1Tensor', 8, repeated=True)
    shape = _messages.IntegerField(9, repeated=True)
    stringVal = _messages.StringField(10, repeated=True)
    structVal = _messages.MessageField('StructValValue', 11)
    tensorVal = _messages.BytesField(12)
    uint64Val = _messages.IntegerField(13, repeated=True, variant=_messages.Variant.UINT64)
    uintVal = _messages.IntegerField(14, repeated=True, variant=_messages.Variant.UINT32)