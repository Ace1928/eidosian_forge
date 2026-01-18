from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportConfig(_messages.Message):
    """Configuration for a Pub/Sub Lite subscription that writes messages to a
  destination. User subscriber clients must not connect to this subscription.

  Enums:
    CurrentStateValueValuesEnum: Output only. The current state of the export,
      which may be different to the desired state due to errors. This field is
      output only.
    DesiredStateValueValuesEnum: The desired state of this export. Setting
      this to values other than `ACTIVE` and `PAUSED` will result in an error.

  Fields:
    currentState: Output only. The current state of the export, which may be
      different to the desired state due to errors. This field is output only.
    deadLetterTopic: Optional. The name of an optional Pub/Sub Lite topic to
      publish messages that can not be exported to the destination. For
      example, the message can not be published to the Pub/Sub service because
      it does not satisfy the constraints documented at
      https://cloud.google.com/pubsub/docs/publisher. Structured like:
      projects/{project_number}/locations/{location}/topics/{topic_id}. Must
      be within the same project and location as the subscription. The topic
      may be changed or removed.
    desiredState: The desired state of this export. Setting this to values
      other than `ACTIVE` and `PAUSED` will result in an error.
    pubsubConfig: Messages are automatically written from the Pub/Sub Lite
      topic associated with this subscription to a Pub/Sub topic.
  """

    class CurrentStateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the export, which may be different
    to the desired state due to errors. This field is output only.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: Messages are being exported.
      PAUSED: Exporting messages is suspended.
      PERMISSION_DENIED: Messages cannot be exported due to permission denied
        errors. Output only.
      NOT_FOUND: Messages cannot be exported due to missing resources. Output
        only.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PAUSED = 2
        PERMISSION_DENIED = 3
        NOT_FOUND = 4

    class DesiredStateValueValuesEnum(_messages.Enum):
        """The desired state of this export. Setting this to values other than
    `ACTIVE` and `PAUSED` will result in an error.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: Messages are being exported.
      PAUSED: Exporting messages is suspended.
      PERMISSION_DENIED: Messages cannot be exported due to permission denied
        errors. Output only.
      NOT_FOUND: Messages cannot be exported due to missing resources. Output
        only.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PAUSED = 2
        PERMISSION_DENIED = 3
        NOT_FOUND = 4
    currentState = _messages.EnumField('CurrentStateValueValuesEnum', 1)
    deadLetterTopic = _messages.StringField(2)
    desiredState = _messages.EnumField('DesiredStateValueValuesEnum', 3)
    pubsubConfig = _messages.MessageField('PubSubConfig', 4)