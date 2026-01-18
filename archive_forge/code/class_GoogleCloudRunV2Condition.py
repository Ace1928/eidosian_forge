from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2Condition(_messages.Message):
    """Defines a status condition for a resource.

  Enums:
    ExecutionReasonValueValuesEnum: Output only. A reason for the execution
      condition.
    ReasonValueValuesEnum: Output only. A common (service-level) reason for
      this condition.
    RevisionReasonValueValuesEnum: Output only. A reason for the revision
      condition.
    SeverityValueValuesEnum: How to interpret failures of this condition, one
      of Error, Warning, Info
    StateValueValuesEnum: State of the condition.

  Fields:
    executionReason: Output only. A reason for the execution condition.
    lastTransitionTime: Last time the condition transitioned from one status
      to another.
    message: Human readable message indicating details about the current
      status.
    reason: Output only. A common (service-level) reason for this condition.
    revisionReason: Output only. A reason for the revision condition.
    severity: How to interpret failures of this condition, one of Error,
      Warning, Info
    state: State of the condition.
    type: type is used to communicate the status of the reconciliation
      process. See also:
      https://github.com/knative/serving/blob/main/docs/spec/errors.md#error-
      conditions-and-reporting Types common to all resources include: *
      "Ready": True when the Resource is ready.
  """

    class ExecutionReasonValueValuesEnum(_messages.Enum):
        """Output only. A reason for the execution condition.

    Values:
      EXECUTION_REASON_UNDEFINED: Default value.
      JOB_STATUS_SERVICE_POLLING_ERROR: Internal system error getting
        execution status. System will retry.
      NON_ZERO_EXIT_CODE: A task reached its retry limit and the last attempt
        failed due to the user container exiting with a non-zero exit code.
      CANCELLED: The execution was cancelled by users.
      CANCELLING: The execution is in the process of being cancelled.
      DELETED: The execution was deleted.
    """
        EXECUTION_REASON_UNDEFINED = 0
        JOB_STATUS_SERVICE_POLLING_ERROR = 1
        NON_ZERO_EXIT_CODE = 2
        CANCELLED = 3
        CANCELLING = 4
        DELETED = 5

    class ReasonValueValuesEnum(_messages.Enum):
        """Output only. A common (service-level) reason for this condition.

    Values:
      COMMON_REASON_UNDEFINED: Default value.
      UNKNOWN: Reason unknown. Further details will be in message.
      REVISION_FAILED: Revision creation process failed.
      PROGRESS_DEADLINE_EXCEEDED: Timed out waiting for completion.
      CONTAINER_MISSING: The container image path is incorrect.
      CONTAINER_PERMISSION_DENIED: Insufficient permissions on the container
        image.
      CONTAINER_IMAGE_UNAUTHORIZED: Container image is not authorized by
        policy.
      CONTAINER_IMAGE_AUTHORIZATION_CHECK_FAILED: Container image policy
        authorization check failed.
      ENCRYPTION_KEY_PERMISSION_DENIED: Insufficient permissions on encryption
        key.
      ENCRYPTION_KEY_CHECK_FAILED: Permission check on encryption key failed.
      SECRETS_ACCESS_CHECK_FAILED: At least one Access check on secrets
        failed.
      WAITING_FOR_OPERATION: Waiting for operation to complete.
      IMMEDIATE_RETRY: System will retry immediately.
      POSTPONED_RETRY: System will retry later; current attempt failed.
      INTERNAL: An internal error occurred. Further information may be in the
        message.
    """
        COMMON_REASON_UNDEFINED = 0
        UNKNOWN = 1
        REVISION_FAILED = 2
        PROGRESS_DEADLINE_EXCEEDED = 3
        CONTAINER_MISSING = 4
        CONTAINER_PERMISSION_DENIED = 5
        CONTAINER_IMAGE_UNAUTHORIZED = 6
        CONTAINER_IMAGE_AUTHORIZATION_CHECK_FAILED = 7
        ENCRYPTION_KEY_PERMISSION_DENIED = 8
        ENCRYPTION_KEY_CHECK_FAILED = 9
        SECRETS_ACCESS_CHECK_FAILED = 10
        WAITING_FOR_OPERATION = 11
        IMMEDIATE_RETRY = 12
        POSTPONED_RETRY = 13
        INTERNAL = 14

    class RevisionReasonValueValuesEnum(_messages.Enum):
        """Output only. A reason for the revision condition.

    Values:
      REVISION_REASON_UNDEFINED: Default value.
      PENDING: Revision in Pending state.
      RESERVE: Revision is in Reserve state.
      RETIRED: Revision is Retired.
      RETIRING: Revision is being retired.
      RECREATING: Revision is being recreated.
      HEALTH_CHECK_CONTAINER_ERROR: There was a health check error.
      CUSTOMIZED_PATH_RESPONSE_PENDING: Health check failed due to user error
        from customized path of the container. System will retry.
      MIN_INSTANCES_NOT_PROVISIONED: A revision with min_instance_count > 0
        was created and is reserved, but it was not configured to serve
        traffic, so it's not live. This can also happen momentarily during
        traffic migration.
      ACTIVE_REVISION_LIMIT_REACHED: The maximum allowed number of active
        revisions has been reached.
      NO_DEPLOYMENT: There was no deployment defined. This value is no longer
        used, but Services created in older versions of the API might contain
        this value.
      HEALTH_CHECK_SKIPPED: A revision's container has no port specified since
        the revision is of a manually scaled service with 0 instance count
      MIN_INSTANCES_WARMING: A revision with min_instance_count > 0 was
        created and is waiting for enough instances to begin a traffic
        migration.
    """
        REVISION_REASON_UNDEFINED = 0
        PENDING = 1
        RESERVE = 2
        RETIRED = 3
        RETIRING = 4
        RECREATING = 5
        HEALTH_CHECK_CONTAINER_ERROR = 6
        CUSTOMIZED_PATH_RESPONSE_PENDING = 7
        MIN_INSTANCES_NOT_PROVISIONED = 8
        ACTIVE_REVISION_LIMIT_REACHED = 9
        NO_DEPLOYMENT = 10
        HEALTH_CHECK_SKIPPED = 11
        MIN_INSTANCES_WARMING = 12

    class SeverityValueValuesEnum(_messages.Enum):
        """How to interpret failures of this condition, one of Error, Warning,
    Info

    Values:
      SEVERITY_UNSPECIFIED: Unspecified severity
      ERROR: Error severity.
      WARNING: Warning severity.
      INFO: Info severity.
    """
        SEVERITY_UNSPECIFIED = 0
        ERROR = 1
        WARNING = 2
        INFO = 3

    class StateValueValuesEnum(_messages.Enum):
        """State of the condition.

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      CONDITION_PENDING: Transient state: Reconciliation has not started yet.
      CONDITION_RECONCILING: Transient state: reconciliation is still in
        progress.
      CONDITION_FAILED: Terminal state: Reconciliation did not succeed.
      CONDITION_SUCCEEDED: Terminal state: Reconciliation completed
        successfully.
    """
        STATE_UNSPECIFIED = 0
        CONDITION_PENDING = 1
        CONDITION_RECONCILING = 2
        CONDITION_FAILED = 3
        CONDITION_SUCCEEDED = 4
    executionReason = _messages.EnumField('ExecutionReasonValueValuesEnum', 1)
    lastTransitionTime = _messages.StringField(2)
    message = _messages.StringField(3)
    reason = _messages.EnumField('ReasonValueValuesEnum', 4)
    revisionReason = _messages.EnumField('RevisionReasonValueValuesEnum', 5)
    severity = _messages.EnumField('SeverityValueValuesEnum', 6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    type = _messages.StringField(8)