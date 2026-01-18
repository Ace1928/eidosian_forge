from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsBackupsCreateRequest(_messages.Message):
    """A FileProjectsLocationsBackupsCreateRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    backupId: Required. The ID to use for the backup. The ID must be unique
      within the specified project and location. This value must start with a
      lowercase letter followed by up to 62 lowercase letters, numbers, or
      hyphens, and cannot end with a hyphen. Values that do not match this
      pattern will trigger an INVALID_ARGUMENT error.
    parent: Required. The backup's project and location, in the format
      `projects/{project_number}/locations/{location}`. In Filestore, backup
      locations map to Google Cloud regions, for example **us-west1**.
  """
    backup = _messages.MessageField('Backup', 1)
    backupId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)