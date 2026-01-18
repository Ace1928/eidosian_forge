from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ColumnTypeValueValuesEnum(_messages.Enum):
    """The data type of a given column.

    Values:
      COLUMN_DATA_TYPE_UNSPECIFIED: Invalid type.
      TYPE_INT64: Encoded as a string in decimal format.
      TYPE_BOOL: Encoded as a boolean "false" or "true".
      TYPE_FLOAT64: Encoded as a number, or string "NaN", "Infinity" or
        "-Infinity".
      TYPE_STRING: Encoded as a string value.
      TYPE_BYTES: Encoded as a base64 string per RFC 4648, section 4.
      TYPE_TIMESTAMP: Encoded as an RFC 3339 timestamp with mandatory "Z" time
        zone string: 1985-04-12T23:20:50.52Z
      TYPE_DATE: Encoded as RFC 3339 full-date format string: 1985-04-12
      TYPE_TIME: Encoded as RFC 3339 partial-time format string: 23:20:50.52
      TYPE_DATETIME: Encoded as RFC 3339 full-date "T" partial-time:
        1985-04-12T23:20:50.52
      TYPE_GEOGRAPHY: Encoded as WKT
      TYPE_NUMERIC: Encoded as a decimal string.
      TYPE_RECORD: Container of ordered fields, each with a type and field
        name.
      TYPE_BIGNUMERIC: Decimal type.
      TYPE_JSON: Json type.
      TYPE_INTERVAL: Interval type.
      TYPE_RANGE_DATE: Range type.
      TYPE_RANGE_DATETIME: Range type.
      TYPE_RANGE_TIMESTAMP: Range type.
    """
    COLUMN_DATA_TYPE_UNSPECIFIED = 0
    TYPE_INT64 = 1
    TYPE_BOOL = 2
    TYPE_FLOAT64 = 3
    TYPE_STRING = 4
    TYPE_BYTES = 5
    TYPE_TIMESTAMP = 6
    TYPE_DATE = 7
    TYPE_TIME = 8
    TYPE_DATETIME = 9
    TYPE_GEOGRAPHY = 10
    TYPE_NUMERIC = 11
    TYPE_RECORD = 12
    TYPE_BIGNUMERIC = 13
    TYPE_JSON = 14
    TYPE_INTERVAL = 15
    TYPE_RANGE_DATE = 16
    TYPE_RANGE_DATETIME = 17
    TYPE_RANGE_TIMESTAMP = 18