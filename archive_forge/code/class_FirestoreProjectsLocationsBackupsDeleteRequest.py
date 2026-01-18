from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsLocationsBackupsDeleteRequest(_messages.Message):
    """A FirestoreProjectsLocationsBackupsDeleteRequest object.

  Fields:
    name: Required. Name of the backup to delete. format is
      `projects/{project}/locations/{location}/backups/{backup}`.
  """
    name = _messages.StringField(1, required=True)