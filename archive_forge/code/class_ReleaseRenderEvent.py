from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReleaseRenderEvent(_messages.Message):
    """Payload proto for "clouddeploy.googleapis.com/release_render" Platform
  Log event that describes the render status change.

  Enums:
    ReleaseRenderStateValueValuesEnum: The state of the release render.
    TypeValueValuesEnum: Type of this notification, e.g. for a release render
      state change event.

  Fields:
    message: Debug message for when a render transition occurs. Provides
      further details as rendering progresses through render states.
    pipelineUid: Unique identifier of the `DeliveryPipeline`.
    release: The name of the release. release_uid is not in this log message
      because we write some of these log messages at release creation time,
      before we've generated the uid.
    releaseRenderState: The state of the release render.
    type: Type of this notification, e.g. for a release render state change
      event.
  """

    class ReleaseRenderStateValueValuesEnum(_messages.Enum):
        """The state of the release render.

    Values:
      RENDER_STATE_UNSPECIFIED: The render state is unspecified.
      SUCCEEDED: All rendering operations have completed successfully.
      FAILED: All rendering operations have completed, and one or more have
        failed.
      IN_PROGRESS: Rendering has started and is not complete.
    """
        RENDER_STATE_UNSPECIFIED = 0
        SUCCEEDED = 1
        FAILED = 2
        IN_PROGRESS = 3

    class TypeValueValuesEnum(_messages.Enum):
        """Type of this notification, e.g. for a release render state change
    event.

    Values:
      TYPE_UNSPECIFIED: Type is unspecified.
      TYPE_PUBSUB_NOTIFICATION_FAILURE: A Pub/Sub notification failed to be
        sent.
      TYPE_RESOURCE_STATE_CHANGE: Resource state changed.
      TYPE_PROCESS_ABORTED: A process aborted.
      TYPE_RESTRICTION_VIOLATED: Restriction check failed.
      TYPE_RESOURCE_DELETED: Resource deleted.
      TYPE_ROLLOUT_UPDATE: Rollout updated.
      TYPE_DEPLOY_POLICY_EVALUATION: Deploy Policy evaluation.
      TYPE_RENDER_STATUES_CHANGE: Deprecated: This field is never used. Use
        release_render log type instead.
    """
        TYPE_UNSPECIFIED = 0
        TYPE_PUBSUB_NOTIFICATION_FAILURE = 1
        TYPE_RESOURCE_STATE_CHANGE = 2
        TYPE_PROCESS_ABORTED = 3
        TYPE_RESTRICTION_VIOLATED = 4
        TYPE_RESOURCE_DELETED = 5
        TYPE_ROLLOUT_UPDATE = 6
        TYPE_DEPLOY_POLICY_EVALUATION = 7
        TYPE_RENDER_STATUES_CHANGE = 8
    message = _messages.StringField(1)
    pipelineUid = _messages.StringField(2)
    release = _messages.StringField(3)
    releaseRenderState = _messages.EnumField('ReleaseRenderStateValueValuesEnum', 4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)