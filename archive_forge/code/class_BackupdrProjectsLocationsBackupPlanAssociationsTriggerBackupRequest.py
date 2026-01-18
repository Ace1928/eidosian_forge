from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupdrProjectsLocationsBackupPlanAssociationsTriggerBackupRequest(_messages.Message):
    """A BackupdrProjectsLocationsBackupPlanAssociationsTriggerBackupRequest
  object.

  Fields:
    name: Required. Name of the backup plan association resource, in the
      format `projects/{project}/locations/{location}/backupPlanAssociations/{
      backupPlanAssociationId}`
    triggerBackupRequest: A TriggerBackupRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    triggerBackupRequest = _messages.MessageField('TriggerBackupRequest', 2)