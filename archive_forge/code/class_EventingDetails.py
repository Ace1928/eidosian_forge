from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventingDetails(_messages.Message):
    """Eventing Details message.

  Enums:
    LaunchStageValueValuesEnum: Output only. Eventing Launch Stage.

  Fields:
    customEventTypes: Output only. Custom Event Types.
    description: Output only. Description.
    documentationLink: Output only. Link to public documentation.
    iconLocation: Output only. Cloud storage location of the icon.
    launchStage: Output only. Eventing Launch Stage.
    name: Output only. Name of the Eventing trigger.
    searchTags: Output only. Array of search keywords.
  """

    class LaunchStageValueValuesEnum(_messages.Enum):
        """Output only. Eventing Launch Stage.

    Values:
      LAUNCH_STAGE_UNSPECIFIED: LAUNCH_STAGE_UNSPECIFIED.
      PREVIEW: PREVIEW.
      GA: GA.
      DEPRECATED: DEPRECATED.
      PRIVATE_PREVIEW: PRIVATE_PREVIEW.
    """
        LAUNCH_STAGE_UNSPECIFIED = 0
        PREVIEW = 1
        GA = 2
        DEPRECATED = 3
        PRIVATE_PREVIEW = 4
    customEventTypes = _messages.BooleanField(1)
    description = _messages.StringField(2)
    documentationLink = _messages.StringField(3)
    iconLocation = _messages.StringField(4)
    launchStage = _messages.EnumField('LaunchStageValueValuesEnum', 5)
    name = _messages.StringField(6)
    searchTags = _messages.StringField(7, repeated=True)