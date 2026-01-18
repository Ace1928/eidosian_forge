from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DefaultSinkConfig(_messages.Message):
    """Describes the custom _Default sink configuration that is used to
  override the built-in _Default sink configuration in newly created resource
  containers, such as projects or folders.

  Enums:
    ModeValueValuesEnum: Required. Determines the behavior to apply to the
      built-in _Default sink inclusion filter.Exclusions are always appended,
      as built-in _Default sinks have no exclusions.

  Fields:
    exclusions: Optional. Specifies the set of exclusions to be added to the
      _Default sink in newly created resource containers.
    filter: Optional. An advanced logs filter
      (https://cloud.google.com/logging/docs/view/advanced-queries). The only
      exported log entries are those that are in the resource owning the sink
      and that match the filter.For
      example:logName="projects/[PROJECT_ID]/logs/[LOG_ID]" AND
      severity>=ERRORTo match all logs, don't add exclusions and use the
      following line as the value of filter:logName:*Cannot be empty or unset
      when the value of mode is OVERWRITE.
    mode: Required. Determines the behavior to apply to the built-in _Default
      sink inclusion filter.Exclusions are always appended, as built-in
      _Default sinks have no exclusions.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Required. Determines the behavior to apply to the built-in _Default
    sink inclusion filter.Exclusions are always appended, as built-in _Default
    sinks have no exclusions.

    Values:
      FILTER_WRITE_MODE_UNSPECIFIED: The filter's write mode is unspecified.
        This mode must not be used.
      APPEND: The contents of filter will be appended to the built-in _Default
        sink filter. Using the append mode with an empty filter will keep the
        sink inclusion filter unchanged.
      OVERWRITE: The contents of filter will overwrite the built-in _Default
        sink filter.
    """
        FILTER_WRITE_MODE_UNSPECIFIED = 0
        APPEND = 1
        OVERWRITE = 2
    exclusions = _messages.MessageField('LogExclusion', 1, repeated=True)
    filter = _messages.StringField(2)
    mode = _messages.EnumField('ModeValueValuesEnum', 3)