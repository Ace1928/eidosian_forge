from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupdrProjectsLocationsBackupVaultsGetRequest(_messages.Message):
    """A BackupdrProjectsLocationsBackupVaultsGetRequest object.

  Fields:
    name: Required. Name of the backupvault store resource name, in the format
      `projects/{project_id}/locations/{location}/backupVaults/{resource_name}
      `
  """
    name = _messages.StringField(1, required=True)