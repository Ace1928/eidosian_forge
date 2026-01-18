from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolloutUpdateEvent(_messages.Message):
    """Payload proto for "clouddeploy.googleapis.com/rollout_update" Platform
  Log event that describes the rollout update event.

  Enums:
    RolloutUpdateTypeValueValuesEnum: The type of the rollout update.
    TypeValueValuesEnum: Type of this notification, e.g. for a rollout update
      event.

  Fields:
    message: Debug message for when a rollout update event occurs.
    pipelineUid: Unique identifier of the pipeline.
    release: The name of the `Release`.
    releaseUid: Unique identifier of the release.
    rollout: The name of the rollout. rollout_uid is not in this log message
      because we write some of these log messages at rollout creation time,
      before we've generated the uid.
    rolloutUpdateType: The type of the rollout update.
    targetId: ID of the target.
    type: Type of this notification, e.g. for a rollout update event.
  """

    class RolloutUpdateTypeValueValuesEnum(_messages.Enum):
        """The type of the rollout update.

    Values:
      ROLLOUT_UPDATE_TYPE_UNSPECIFIED: Rollout update type unspecified.
      PENDING: rollout state updated to pending.
      PENDING_RELEASE: Rollout state updated to pending release.
      IN_PROGRESS: Rollout state updated to in progress.
      CANCELLING: Rollout state updated to cancelling.
      CANCELLED: Rollout state updated to cancelled.
      HALTED: Rollout state updated to halted.
      SUCCEEDED: Rollout state updated to succeeded.
      FAILED: Rollout state updated to failed.
      APPROVAL_REQUIRED: Rollout requires approval.
      APPROVED: Rollout has been approved.
      REJECTED: Rollout has been rejected.
      ADVANCE_REQUIRED: Rollout requires advance to the next phase.
      ADVANCED: Rollout has been advanced.
    """
        ROLLOUT_UPDATE_TYPE_UNSPECIFIED = 0
        PENDING = 1
        PENDING_RELEASE = 2
        IN_PROGRESS = 3
        CANCELLING = 4
        CANCELLED = 5
        HALTED = 6
        SUCCEEDED = 7
        FAILED = 8
        APPROVAL_REQUIRED = 9
        APPROVED = 10
        REJECTED = 11
        ADVANCE_REQUIRED = 12
        ADVANCED = 13

    class TypeValueValuesEnum(_messages.Enum):
        """Type of this notification, e.g. for a rollout update event.

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
    releaseUid = _messages.StringField(4)
    rollout = _messages.StringField(5)
    rolloutUpdateType = _messages.EnumField('RolloutUpdateTypeValueValuesEnum', 6)
    targetId = _messages.StringField(7)
    type = _messages.EnumField('TypeValueValuesEnum', 8)