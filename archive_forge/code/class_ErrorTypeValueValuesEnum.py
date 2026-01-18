from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorTypeValueValuesEnum(_messages.Enum):
    """The error type of this record.

    Values:
      ERROR_TYPE_UNSPECIFIED: Default, shall not be used.
      EMPTY_LINE: The record is empty.
      INVALID_JSON_SYNTAX: Invalid json format.
      INVALID_CSV_SYNTAX: Invalid csv format.
      INVALID_AVRO_SYNTAX: Invalid avro format.
      INVALID_EMBEDDING_ID: The embedding id is not valid.
      EMBEDDING_SIZE_MISMATCH: The size of the embedding vectors does not
        match with the specified dimension.
      NAMESPACE_MISSING: The `namespace` field is missing.
      PARSING_ERROR: Generic catch-all error. Only used for validation failure
        where the root cause cannot be easily retrieved programmatically.
      DUPLICATE_NAMESPACE: There are multiple restricts with the same
        `namespace` value.
      OP_IN_DATAPOINT: Numeric restrict has operator specified in datapoint.
      MULTIPLE_VALUES: Numeric restrict has multiple values specified.
      INVALID_NUMERIC_VALUE: Numeric restrict has invalid numeric value
        specified.
      INVALID_ENCODING: File is not in UTF_8 format.
    """
    ERROR_TYPE_UNSPECIFIED = 0
    EMPTY_LINE = 1
    INVALID_JSON_SYNTAX = 2
    INVALID_CSV_SYNTAX = 3
    INVALID_AVRO_SYNTAX = 4
    INVALID_EMBEDDING_ID = 5
    EMBEDDING_SIZE_MISMATCH = 6
    NAMESPACE_MISSING = 7
    PARSING_ERROR = 8
    DUPLICATE_NAMESPACE = 9
    OP_IN_DATAPOINT = 10
    MULTIPLE_VALUES = 11
    INVALID_NUMERIC_VALUE = 12
    INVALID_ENCODING = 13