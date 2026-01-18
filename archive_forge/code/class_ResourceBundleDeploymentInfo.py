from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceBundleDeploymentInfo(_messages.Message):
    """ResourceBundleDeploymentInfo represents the status of a resource bundle
  deployment.

  Enums:
    DeletionPropagationPolicyValueValuesEnum: Output only. Deletion
      propagation policy.
    StateValueValuesEnum: Output only. State of synchronization of the
      resource bundle deployment.

  Fields:
    deletionPropagationPolicy: Output only. Deletion propagation policy.
    messages: Output only. Messages convey additional information related to
      the package deployment. For example, in case of an error, indicate the
      reason for the error. In case of a pending deployment, reason for why
      the deployment of new release is pending.
    reconciliationEndTime: Output only. Timestamp when reconciliation ends.
    reconciliationStartTime: Output only. Timestamp when reconciliation
      starts.
    release: Output only. Refers to a package release.
    state: Output only. State of synchronization of the resource bundle
      deployment.
    variant: Output only. Refers to a package variant.
    version: Output only. Refers to a package version.
  """

    class DeletionPropagationPolicyValueValuesEnum(_messages.Enum):
        """Output only. Deletion propagation policy.

    Values:
      DELETION_PROPAGATION_POLICY_UNSPECIFIED: Unspecified deletion
        propagation policy. Defaults to FOREGROUND.
      FOREGROUND: Foreground deletion propagation policy. Any resources synced
        to the cluster will be deleted.
      ORPHAN: Orphan deletion propagation policy. Any resources synced to the
        cluster will be abandoned.
    """
        DELETION_PROPAGATION_POLICY_UNSPECIFIED = 0
        FOREGROUND = 1
        ORPHAN = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of synchronization of the resource bundle
    deployment.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      RECONCILING: Reconciling state.
      STALLED: Stalled state.
      SYNCED: Synced state.
      PENDING: Pending state.
      ERROR: Error state.
      WAITING: Waiting state.
      DELETION_PENDING: Deletion pending state.
      DELETING: Deleting state.
      DELETED: Deleted state.
      INITIATED: Initiated state.
      ABORTED: Aborted state.
    """
        STATE_UNSPECIFIED = 0
        RECONCILING = 1
        STALLED = 2
        SYNCED = 3
        PENDING = 4
        ERROR = 5
        WAITING = 6
        DELETION_PENDING = 7
        DELETING = 8
        DELETED = 9
        INITIATED = 10
        ABORTED = 11
    deletionPropagationPolicy = _messages.EnumField('DeletionPropagationPolicyValueValuesEnum', 1)
    messages = _messages.StringField(2, repeated=True)
    reconciliationEndTime = _messages.StringField(3)
    reconciliationStartTime = _messages.StringField(4)
    release = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    variant = _messages.StringField(7)
    version = _messages.StringField(8)