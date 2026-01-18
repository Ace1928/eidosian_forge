from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowTrigger(_messages.Message):
    """Workflow trigger within a Workflow.

  Enums:
    EventTypeValueValuesEnum: Optional. The type of the events the
      WorkflowTrigger accepts.
    StatusValueValuesEnum: Output only. The status of the WorkflowTrigger.

  Fields:
    createTime: Output only. Creation time of the WorkflowTrigger.
    custom: The CEL filters that triggers the Workflow.
    eventType: Optional. The type of the events the WorkflowTrigger accepts.
    gitRef: Optional. The Git ref matching the SCM repo branch/tag.
    id: Immutable. id given by the users to the Workflow.
    params: List of parameters associated with the WorkflowTrigger.
    pullRequest: Optional. The Pull request role and comment that triggers the
      Workflow.
    source: The event source the WorkflowTrigger listens to.
    status: Output only. The status of the WorkflowTrigger.
    statusMessage: Output only. The reason why WorkflowTrigger is deactivated.
    updateTime: Output only. Update time of the WorkflowTrigger.
    uuid: Output only. The internal id of the WorkflowTrigger.
    webhookSecret: The webhook secret resource.
  """

    class EventTypeValueValuesEnum(_messages.Enum):
        """Optional. The type of the events the WorkflowTrigger accepts.

    Values:
      EVENTTYPE_UNSPECIFIED: Default to ALL.
      ALL: All events.
      PULL_REQUEST: PR events.
      PUSH_BRANCH: Push to branch events.
      PUSH_TAG: Push to tag events.
    """
        EVENTTYPE_UNSPECIFIED = 0
        ALL = 1
        PULL_REQUEST = 2
        PUSH_BRANCH = 3
        PUSH_TAG = 4

    class StatusValueValuesEnum(_messages.Enum):
        """Output only. The status of the WorkflowTrigger.

    Values:
      STATUS_UNSPECIFIED: Defaults to ACTIVE.
      ACTIVE: WorkflowTrigger is active.
      DEACTIVATED: WorkflowTrigger is deactivated.
    """
        STATUS_UNSPECIFIED = 0
        ACTIVE = 1
        DEACTIVATED = 2
    createTime = _messages.StringField(1)
    custom = _messages.MessageField('CEL', 2, repeated=True)
    eventType = _messages.EnumField('EventTypeValueValuesEnum', 3)
    gitRef = _messages.MessageField('GitRef', 4)
    id = _messages.StringField(5)
    params = _messages.MessageField('Param', 6, repeated=True)
    pullRequest = _messages.MessageField('PullRequest', 7)
    source = _messages.MessageField('EventSource', 8)
    status = _messages.EnumField('StatusValueValuesEnum', 9)
    statusMessage = _messages.StringField(10)
    updateTime = _messages.StringField(11)
    uuid = _messages.StringField(12)
    webhookSecret = _messages.MessageField('WebhookSecret', 13)