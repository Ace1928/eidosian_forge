from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StandardSqlDataType(_messages.Message):
    """The data type of a variable such as a function argument. Examples
  include: * INT64: `{"typeKind": "INT64"}` * ARRAY: { "typeKind": "ARRAY",
  "arrayElementType": {"typeKind": "STRING"} } * STRUCT>: { "typeKind":
  "STRUCT", "structType": { "fields": [ { "name": "x", "type": {"typeKind":
  "STRING"} }, { "name": "y", "type": { "typeKind": "ARRAY",
  "arrayElementType": {"typeKind": "DATE"} } } ] } }

  Enums:
    TypeKindValueValuesEnum: Required. The top level type of this field. Can
      be any GoogleSQL data type (e.g., "INT64", "DATE", "ARRAY").

  Fields:
    arrayElementType: The type of the array's elements, if type_kind =
      "ARRAY".
    rangeElementType: The type of the range's elements, if type_kind =
      "RANGE".
    structType: The fields of this struct, in order, if type_kind = "STRUCT".
    typeKind: Required. The top level type of this field. Can be any GoogleSQL
      data type (e.g., "INT64", "DATE", "ARRAY").
  """

    class TypeKindValueValuesEnum(_messages.Enum):
        """Required. The top level type of this field. Can be any GoogleSQL data
    type (e.g., "INT64", "DATE", "ARRAY").

    Values:
      TYPE_KIND_UNSPECIFIED: Invalid type.
      INT64: Encoded as a string in decimal format.
      BOOL: Encoded as a boolean "false" or "true".
      FLOAT64: Encoded as a number, or string "NaN", "Infinity" or
        "-Infinity".
      STRING: Encoded as a string value.
      BYTES: Encoded as a base64 string per RFC 4648, section 4.
      TIMESTAMP: Encoded as an RFC 3339 timestamp with mandatory "Z" time zone
        string: 1985-04-12T23:20:50.52Z
      DATE: Encoded as RFC 3339 full-date format string: 1985-04-12
      TIME: Encoded as RFC 3339 partial-time format string: 23:20:50.52
      DATETIME: Encoded as RFC 3339 full-date "T" partial-time:
        1985-04-12T23:20:50.52
      INTERVAL: Encoded as fully qualified 3 part: 0-5 15 2:30:45.6
      GEOGRAPHY: Encoded as WKT
      NUMERIC: Encoded as a decimal string.
      BIGNUMERIC: Encoded as a decimal string.
      JSON: Encoded as a string.
      ARRAY: Encoded as a list with types matching Type.array_type.
      STRUCT: Encoded as a list with fields of type Type.struct_type[i]. List
        is used because a JSON object cannot have duplicate field names.
      RANGE: Encoded as a pair with types matching range_element_type. Pairs
        must begin with "[", end with ")", and be separated by ", ".
    """
        TYPE_KIND_UNSPECIFIED = 0
        INT64 = 1
        BOOL = 2
        FLOAT64 = 3
        STRING = 4
        BYTES = 5
        TIMESTAMP = 6
        DATE = 7
        TIME = 8
        DATETIME = 9
        INTERVAL = 10
        GEOGRAPHY = 11
        NUMERIC = 12
        BIGNUMERIC = 13
        JSON = 14
        ARRAY = 15
        STRUCT = 16
        RANGE = 17
    arrayElementType = _messages.MessageField('StandardSqlDataType', 1)
    rangeElementType = _messages.MessageField('StandardSqlDataType', 2)
    structType = _messages.MessageField('StandardSqlStructType', 3)
    typeKind = _messages.EnumField('TypeKindValueValuesEnum', 4)