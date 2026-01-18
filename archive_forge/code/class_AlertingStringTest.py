from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlertingStringTest(_messages.Message):
    """A test that compares a string column against a string to match. NOTE:
  StringTest is not yet supported.

  Enums:
    ComparisonValueValuesEnum: Required. The comparison operator to use.

  Fields:
    column: Required. The column that contains the strings we want to search
      on.
    comparison: Required. The comparison operator to use.
    pattern: Required. The string or regular expression which is compared to
      the value in the column.
    trigger: Optional. The number/percent of rows that must match in order for
      the result set (partition set) to be considered in violation. If
      unspecified, then the result set (partition set) will be in violation if
      a single row matches.
  """

    class ComparisonValueValuesEnum(_messages.Enum):
        """Required. The comparison operator to use.

    Values:
      STRING_COMPARISON_TYPE_UNSPECIFIED: No string comparison specified,
        should never happen.
      STRING_COMPARISON_MATCH: String column must equal the pattern.
      STRING_COMPARISON_NOT_MATCH: String column must not equal the pattern.
      STRING_COMPARISON_CONTAINS: String contains contains the pattern as a
        substring.
      STRING_COMPARISON_NOT_CONTAINS: String column does not contain the
        pattern as a substring.
      STRING_COMPARISON_REGEX_MATCH: Regular expression pattern found in
        string column.
      STRING_COMPARISON_REGEX_NOT_MATCH: Regular expression pattern not found
        in string column.
    """
        STRING_COMPARISON_TYPE_UNSPECIFIED = 0
        STRING_COMPARISON_MATCH = 1
        STRING_COMPARISON_NOT_MATCH = 2
        STRING_COMPARISON_CONTAINS = 3
        STRING_COMPARISON_NOT_CONTAINS = 4
        STRING_COMPARISON_REGEX_MATCH = 5
        STRING_COMPARISON_REGEX_NOT_MATCH = 6
    column = _messages.StringField(1)
    comparison = _messages.EnumField('ComparisonValueValuesEnum', 2)
    pattern = _messages.StringField(3)
    trigger = _messages.MessageField('AlertingTrigger', 4)