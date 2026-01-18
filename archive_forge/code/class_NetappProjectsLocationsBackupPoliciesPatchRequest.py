from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsBackupPoliciesPatchRequest(_messages.Message):
    """A NetappProjectsLocationsBackupPoliciesPatchRequest object.

  Fields:
    backupPolicy: A BackupPolicy resource to be passed as the request body.
    name: Identifier. The resource name of the backup policy. Format: `project
      s/{project_id}/locations/{location}/backupPolicies/{backup_policy_id}`.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the Backup Policy resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. If the user
      does not provide a mask then all fields will be overwritten.
  """
    backupPolicy = _messages.MessageField('BackupPolicy', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)