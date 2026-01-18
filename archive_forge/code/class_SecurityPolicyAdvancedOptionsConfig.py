from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyAdvancedOptionsConfig(_messages.Message):
    """A SecurityPolicyAdvancedOptionsConfig object.

  Enums:
    JsonParsingValueValuesEnum:
    LogLevelValueValuesEnum:

  Fields:
    jsonCustomConfig: Custom configuration to apply the JSON parsing. Only
      applicable when json_parsing is set to STANDARD.
    jsonParsing: A JsonParsingValueValuesEnum attribute.
    logLevel: A LogLevelValueValuesEnum attribute.
    userIpRequestHeaders: An optional list of case-insensitive request header
      names to use for resolving the callers client IP address.
  """

    class JsonParsingValueValuesEnum(_messages.Enum):
        """JsonParsingValueValuesEnum enum type.

    Values:
      DISABLED: <no description>
      STANDARD: <no description>
      STANDARD_WITH_GRAPHQL: <no description>
    """
        DISABLED = 0
        STANDARD = 1
        STANDARD_WITH_GRAPHQL = 2

    class LogLevelValueValuesEnum(_messages.Enum):
        """LogLevelValueValuesEnum enum type.

    Values:
      NORMAL: <no description>
      VERBOSE: <no description>
    """
        NORMAL = 0
        VERBOSE = 1
    jsonCustomConfig = _messages.MessageField('SecurityPolicyAdvancedOptionsConfigJsonCustomConfig', 1)
    jsonParsing = _messages.EnumField('JsonParsingValueValuesEnum', 2)
    logLevel = _messages.EnumField('LogLevelValueValuesEnum', 3)
    userIpRequestHeaders = _messages.StringField(4, repeated=True)