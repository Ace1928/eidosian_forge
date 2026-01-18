from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JsonPathMatcher(_messages.Message):
    """Information needed to perform a JSONPath content match. Used for
  ContentMatcherOption::MATCHES_JSON_PATH and
  ContentMatcherOption::NOT_MATCHES_JSON_PATH.

  Enums:
    JsonMatcherValueValuesEnum: The type of JSONPath match that will be
      applied to the JSON output (ContentMatcher.content)

  Fields:
    jsonMatcher: The type of JSONPath match that will be applied to the JSON
      output (ContentMatcher.content)
    jsonPath: JSONPath within the response output pointing to the expected
      ContentMatcher::content to match against.
  """

    class JsonMatcherValueValuesEnum(_messages.Enum):
        """The type of JSONPath match that will be applied to the JSON output
    (ContentMatcher.content)

    Values:
      JSON_PATH_MATCHER_OPTION_UNSPECIFIED: No JSONPath matcher type specified
        (not valid).
      EXACT_MATCH: Selects 'exact string' matching. The match succeeds if the
        content at the json_path within the output is exactly the same as the
        content string.
      REGEX_MATCH: Selects regular-expression matching. The match succeeds if
        the content at the json_path within the output matches the regular
        expression specified in the content string.
    """
        JSON_PATH_MATCHER_OPTION_UNSPECIFIED = 0
        EXACT_MATCH = 1
        REGEX_MATCH = 2
    jsonMatcher = _messages.EnumField('JsonMatcherValueValuesEnum', 1)
    jsonPath = _messages.StringField(2)