from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutomationRunEvent(_messages.Message):
    """Payload proto for "clouddeploy.googleapis.com/automation_run" Platform
  Log event that describes the AutomationRun related events.

  Enums:
    TypeValueValuesEnum: Type of this notification, e.g. for a Pub/Sub
      failure.

  Fields:
    automationId: Identifier of the `Automation`.
    automationRun: The name of the `AutomationRun`.
    destinationTargetId: ID of the `Target` to which the `AutomationRun` is
      created.
    message: Debug message for when there is an update on the AutomationRun.
      Provides further details about the resource creation or state change.
    pipelineUid: Unique identifier of the `DeliveryPipeline`.
    ruleId: Identifier of the `Automation` rule.
    type: Type of this notification, e.g. for a Pub/Sub failure.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of this notification, e.g. for a Pub/Sub failure.

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
    automationId = _messages.StringField(1)
    automationRun = _messages.StringField(2)
    destinationTargetId = _messages.StringField(3)
    message = _messages.StringField(4)
    pipelineUid = _messages.StringField(5)
    ruleId = _messages.StringField(6)
    type = _messages.EnumField('TypeValueValuesEnum', 7)