from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcpBackupConfig(_messages.Message):
    """GcpBackupConfig captures the Backup configuration details for GCP
  resources. All GCP resources regardless of type are protected with backup
  plan associations.

  Fields:
    backupPlan: The name of the backup plan.
    backupPlanAssociation: The name of the backup plan association.
    backupPlanDescription: The description of the backup plan.
    backupPlanRules: The names of the backup plan rules which point to this
      backupvault
  """
    backupPlan = _messages.StringField(1)
    backupPlanAssociation = _messages.StringField(2)
    backupPlanDescription = _messages.StringField(3)
    backupPlanRules = _messages.StringField(4, repeated=True)