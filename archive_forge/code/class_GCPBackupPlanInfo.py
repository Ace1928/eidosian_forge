from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GCPBackupPlanInfo(_messages.Message):
    """GCPBackupPlanInfo captures the plan configuration details of GCP
  resources at the time of backup.

  Fields:
    backupPlan: Resource name of backup plan by which workload is protected at
      the time of the backup. Format:
      projects/{project}/locations/{location}/backupPlans/{backupPlanId}
    backupPlanRuleId: The rule id of the backup plan which triggered this
      backup in case of scheduled backup or used for
  """
    backupPlan = _messages.StringField(1)
    backupPlanRuleId = _messages.StringField(2)