from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContentMatcher(_messages.Message):
    """Optional. Used to perform content matching. This allows matching based
  on substrings and regular expressions, together with their negations. Only
  the first 4 MB of an HTTP or HTTPS check's response (and the first 1 MB of a
  TCP check's response) are examined for purposes of content matching.

  Enums:
    MatcherValueValuesEnum: The type of content matcher that will be applied
      to the server output, compared to the content string when the check is
      run.

  Fields:
    content: String, regex or JSON content to match. Maximum 1024 bytes. An
      empty content string indicates no content matching is to be performed.
    jsonPathMatcher: Matcher information for MATCHES_JSON_PATH and
      NOT_MATCHES_JSON_PATH
    matcher: The type of content matcher that will be applied to the server
      output, compared to the content string when the check is run.
  """

    class MatcherValueValuesEnum(_messages.Enum):
        """The type of content matcher that will be applied to the server output,
    compared to the content string when the check is run.

    Values:
      CONTENT_MATCHER_OPTION_UNSPECIFIED: No content matcher type specified
        (maintained for backward compatibility, but deprecated for future
        use). Treated as CONTAINS_STRING.
      CONTAINS_STRING: Selects substring matching. The match succeeds if the
        output contains the content string. This is the default value for
        checks without a matcher option, or where the value of matcher is
        CONTENT_MATCHER_OPTION_UNSPECIFIED.
      NOT_CONTAINS_STRING: Selects negation of substring matching. The match
        succeeds if the output does NOT contain the content string.
      MATCHES_REGEX: Selects regular-expression matching. The match succeeds
        if the output matches the regular expression specified in the content
        string. Regex matching is only supported for HTTP/HTTPS checks.
      NOT_MATCHES_REGEX: Selects negation of regular-expression matching. The
        match succeeds if the output does NOT match the regular expression
        specified in the content string. Regex matching is only supported for
        HTTP/HTTPS checks.
      MATCHES_JSON_PATH: Selects JSONPath matching. See JsonPathMatcher for
        details on when the match succeeds. JSONPath matching is only
        supported for HTTP/HTTPS checks.
      NOT_MATCHES_JSON_PATH: Selects JSONPath matching. See JsonPathMatcher
        for details on when the match succeeds. Succeeds when output does NOT
        match as specified. JSONPath is only supported for HTTP/HTTPS checks.
    """
        CONTENT_MATCHER_OPTION_UNSPECIFIED = 0
        CONTAINS_STRING = 1
        NOT_CONTAINS_STRING = 2
        MATCHES_REGEX = 3
        NOT_MATCHES_REGEX = 4
        MATCHES_JSON_PATH = 5
        NOT_MATCHES_JSON_PATH = 6
    content = _messages.StringField(1)
    jsonPathMatcher = _messages.MessageField('JsonPathMatcher', 2)
    matcher = _messages.EnumField('MatcherValueValuesEnum', 3)