from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupPlanAssociation(_messages.Message):
    """A BackupPlanAssociation represents a single BackupPlanAssociation which
  contains details like workload, backup plan etc

  Enums:
    StateValueValuesEnum: Output only. The BackupPlanAssociation resource
      state.

  Fields:
    backupCollectionId: Output only. TODO b/325560313: Deprecated and field
      will be removed after UI integration change. Output only. This will
      return the relative url of the AGM application object which will be used
      to redirect to AGM UI from the pantheon to show a list of backups.
    backupPlan: Required. Resource name of backup plan which needs to be
      applied on workload. Format:
      projects/{project}/locations/{location}/backupPlans/{backupPlanId}
    createTime: Output only. The time when the instance was created.
    name: Output only. The resource name of BackupPlanAssociation in below
      format Format : projects/{project}/locations/{location}/backupPlanAssoci
      ations/{backupPlanAssociationId}
    resource: Required. Immutable. Resource name of workload on which
      backupplan is applied
    resourceType: Output only. Output Only. Resource type of workload on which
      backupplan is applied
    rulesConfigInfo: Output only. The config info related to backup rules.
    scopeId: Output only. TODO b/325560313: Deprecated and field will be
      removed after UI integration change. This will return the scope with
      which this backup plan association is associated with.
    state: Output only. The BackupPlanAssociation resource state.
    updateTime: Output only. The time when the instance was updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The BackupPlanAssociation resource state.

    Values:
      STATE_UNSPECIFIED: State not set.
      CREATING: The resource is being created.
      ACTIVE: The resource has been created and is fully usable.
      DELETING: The resource is being deleted.
      READY: TODO b/325560313: Deprecated and field will be removed after UI
        integration change. The resource has been created and is fully usable.
      INACTIVE: The resource has been created but is not usable.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        READY = 4
        INACTIVE = 5
    backupCollectionId = _messages.StringField(1)
    backupPlan = _messages.StringField(2)
    createTime = _messages.StringField(3)
    name = _messages.StringField(4)
    resource = _messages.StringField(5)
    resourceType = _messages.StringField(6)
    rulesConfigInfo = _messages.MessageField('RuleConfigInfo', 7, repeated=True)
    scopeId = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    updateTime = _messages.StringField(10)